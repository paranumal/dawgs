/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "ogs.hpp"
#include "ogs/ogsKernels.hpp"
#include "ogs/ogsGatherScatter.hpp"
#include "ogs/ogsExchange.hpp"

namespace ogs {

void ogsCrystalRouter_t::Start(occa::memory& o_v, bool gpu_aware, bool overlap){

  const size_t Nbytes = sizeof(dfloat);
  reallocOccaBuffer(Nbytes);

  occa::device &device = platform.device;
  occa::stream currentStream = device.getStream();

  if (gatherHalo->Nrows) {
    //if overlapping the halo kernels, switch streams
    if (overlap)
      device.setStream(dataStream);

    // assemble mpi send buffer by gathering halo nodes
    gatherHalo->Apply(o_haloBuf, o_v);

    //if not overlapping, wait for kernel to finish on default stream
    if (!overlap)
      device.finish();

    if (!gpu_aware) {
      //switch streams to overlap data movement
      device.setStream(dataStream);
      o_haloBuf.copyTo(haloBuf, gatherHalo->Nrows*Nbytes, 0, "async: true");
    }

    device.setStream(currentStream);
  }
}


void ogsCrystalRouter_t::Finish(occa::memory& o_v, bool gpu_aware, bool overlap){

  const size_t Nbytes = sizeof(dfloat);
  occa::device &device = platform.device;

  occa::stream currentStream = device.getStream();

  char *sBufPtr, *rBufPtr;
  if (gpu_aware) { //device pointer
    sBufPtr = (char*)o_sendBuf.ptr();
    rBufPtr = (char*)o_recvBuf.ptr();
  } else { //host pointer
    sBufPtr = (char*)sendBuf;
    rBufPtr = (char*)recvBuf;
  }

  //the intermediate kernels are always overlapped with the default stream
  device.setStream(dataStream);

  for (int k=0;k<Nlevels;k++) {

    //post recvs
    if (levels[k].Nmsg>0)
      MPI_Irecv(rBufPtr, levels[k].Nrecv0, MPI_DFLOAT,
                levels[k].partner, levels[k].partner, comm, request+1);
    if (levels[k].Nmsg==2)
      MPI_Irecv(rBufPtr+levels[k].Nrecv0*Nbytes,
                levels[k].Nrecv1, MPI_DFLOAT,
                rank-1, rank-1, comm, request+2);

    //assemble send buffer
    if (gpu_aware) {
      if (levels[k].Nsend) {
        extractKernel(levels[k].Nsend, levels[k].o_sendIds,
                      o_haloBuf, o_sendBuf);
        device.finish();
      }
    } else {
      for (int n=0;n<levels[k].Nsend;n++) {
        ((dfloat*)sBufPtr)[n] = ((dfloat*)haloBuf)[levels[k].sendIds[n]];
      }
    }

    //post send
    std::this_thread::sleep_for(std::chrono::nanoseconds(overhead));
    MPI_Isend(sBufPtr, levels[k].Nsend, MPI_DFLOAT,
              levels[k].partner, rank, comm, request+0);

    if (k==0) {
      //zero extra section in halo buffer
      if (gpu_aware) {
        if (NhaloExt-Nhalo) {
          const dfloat zero = 0.0;
          setKernel(NhaloExt-Nhalo, zero, o_haloBuf+Nhalo*Nbytes);
        }
      } else {
        for (int n=Nhalo;n<NhaloExt;n++)
          ((dfloat*)haloBuf)[n] = 0.0;
      }
    }

    MPI_Waitall(levels[k].Nmsg+1, request, status);

    //Scatter the recv buffer into the haloBuffer
    if (levels[k].Nmsg>0) {
      if (gpu_aware) {
        if (levels[k].Nrecv0) {
          injectKernel(levels[k].Nrecv0, levels[k].o_recvIds0,
                       o_recvBuf, o_haloBuf);
        }
      } else {
        for (int n=0;n<levels[k].Nrecv0;n++) {
          ((dfloat*)haloBuf)[levels[k].recvIds0[n]] += ((dfloat*)rBufPtr)[n];
        }
      }
    }
    if (levels[k].Nmsg==2) {
      if (gpu_aware) {
        if (levels[k].Nrecv1) {
          injectKernel(levels[k].Nrecv1, levels[k].o_recvIds1,
                       o_recvBuf + levels[k].Nrecv0*Nbytes, o_haloBuf);
        }
      } else {
        for (int n=0;n<levels[k].Nrecv1;n++) {
          ((dfloat*)haloBuf)[levels[k].recvIds1[n]] += ((dfloat*)rBufPtr)[n+levels[k].Nrecv0];
        }
      }
    }
  }

  if (scatterHalo->Ncols) {
    if (!gpu_aware) {
      // copy recv back to device
      device.setStream(dataStream);
      o_haloBuf.copyFrom(haloBuf, scatterHalo->Ncols*Nbytes, 0, "async: true");
      if (!overlap) device.finish(); //wait for transfer to finish if not overlapping halo kernel
    }

    //if overlapping the halo kernels, switch streams
    if (overlap) {
      device.setStream(dataStream);
    } else {
      device.finish();
      device.setStream(currentStream);
    }

    scatterHalo->Apply(o_v, o_haloBuf);

    if (overlap) { //if overlapping halo kernels wait for kernel to finish
      device.finish();
    }
  }

  device.setStream(currentStream);
}

/*
 *Crystal Router performs the needed MPI communcation via recursive
 * folding of a hypercube. Consider a set of NP ranks. We select a
 * pivot point n_half=(NP+1)/2, and pair all ranks r<n_half (called
 * lo half) the with ranks r>=n_half (called the hi half), as follows
 *
 *                0 <--> NP-1
 *                1 <--> NP-2
 *                2 <--> NP-3
 *                  * * *
 *         n_half-2 <--> NP-n_half+1
 *         n_half-1 <--> NP-n_half
 *
 * The communication can then be summarized thusly: if a rank in the lo
 * half has data needed by *any* rank in the hi half, it sends this data
 * to its hi partner, and analogously for ranks in the hi half. Each rank
 * therefore sends/receives a single message to/from its partner.
 *
 * The communication then proceeds recursively, applying the same folding
 * proceedure to the lo and hi halves seperately, and stopping when the size
 * of the local NP reaches 1.
 *
 * In the case where NP is odd, n_half-1 == NP-n_half and rank n_half-1 has
 * no partner to communicate with. In this case, we assign rank r to the
 * lo half of ranks, and rank n_half-1 sends its data to rank n_half (and
 * receives no message, as rank n_half-2 is receiving all rank n_half's data).

 * To perform the Crystal Router exchange, each rank gathers its halo nodes to
 * a coalesced buffer. At each step in the crystal router, a send buffer is
 * gathered from this buffer and sent to the rank's partner. Simultaneously, a
 * buffer is received from the rank's partner. This receive buffer is scattered
 * and added into the coalesced halo buffer. After all commincation is complete
 * the halo nodes are scattered back to the output array.
 */

ogsCrystalRouter_t::ogsCrystalRouter_t(dlong recvN,
                             parallelNode_t* recvNodes,
                             dlong NgatherLocal,
                             ogsGather_t *_gatherHalo,
                             dlong *indexMap,
                             MPI_Comm _comm,
                             platform_t &_platform):
  platform(_platform), comm(_comm) {

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::sort(recvNodes, recvNodes+recvN,
            [](const parallelNode_t& a, const parallelNode_t& b) {
              if(abs(a.baseId) < abs(b.baseId)) return true; //group by abs(baseId)
              if(abs(a.baseId) > abs(b.baseId)) return false;

              return a.baseId < b.baseId; //positive ids first
            });

  gatherHalo = _gatherHalo;
  Nhalo    = _gatherHalo->Nrows;
  NhaloExt = _gatherHalo->Nrows;

  dlong N = recvN;
  parallelNode_t *nodes = recvNodes;

  //first count how many levels we need
  Nlevels = 0;
  int np = size;
  int np_offset=0;
  while (np>1) {
    int np_half = (np+1)/2;
    int r_half = np_half + np_offset;

    int is_lo = (rank<r_half) ? 1 : 0;

    //Shrink the size of the hypercube
    if (is_lo) {
      np = np_half;
    } else {
      np -= np_half;
      np_offset = r_half;
    }
    Nlevels++;
  }
  levels = new crLevel[Nlevels];

  //Now build the levels
  Nlevels = 0;
  np = size;
  np_offset=0;
  while (np>1) {
    int np_half = (np+1)/2;
    int r_half = np_half + np_offset;

    int is_lo = (rank<r_half) ? 1 : 0;

    int partner = np-1-(rank-np_offset)+np_offset;
    int Nmsg=1;
    if (partner==rank) {
      partner=r_half;
      Nmsg=0;
    }
    if (np&1 && rank==r_half) {
      Nmsg=2;
    }
    levels[Nlevels].partner = partner;
    levels[Nlevels].Nmsg = Nmsg;

    //count lo/hi nodes
    dlong Nlo=0, Nhi=0;
    for (dlong n=0;n<N;n++) {
      if (nodes[n].rank<r_half)
        Nlo++;
      else
        Nhi++;
    }

    int Nsend=(is_lo) ? Nhi : Nlo;

    MPI_Isend(&Nsend, 1, MPI_INT, partner, rank, comm, request+0);

    int Nrecv0=0, Nrecv1=0;
    if (Nmsg>0)
      MPI_Irecv(&Nrecv0, 1, MPI_INT, partner, partner, comm, request+1);
    if (Nmsg==2)
      MPI_Irecv(&Nrecv1, 1, MPI_INT, r_half-1, r_half-1, comm, request+2);

    MPI_Waitall(Nmsg+1, request, status);

    int Nrecv = Nrecv0+Nrecv1;

    //make room for the nodes we'll recv
    if (is_lo) Nlo+=Nrecv;
    else       Nhi+=Nrecv;

    //split node list in two
    parallelNode_t *loNodes = (parallelNode_t *) malloc(Nlo*sizeof(parallelNode_t));
    parallelNode_t *hiNodes = (parallelNode_t *) malloc(Nhi*sizeof(parallelNode_t));

    Nlo=0, Nhi=0;
    for (dlong n=0;n<N;n++) {
      if (nodes[n].rank<r_half)
        loNodes[Nlo++] = nodes[n];
      else
        hiNodes[Nhi++] = nodes[n];
    }

    if (np!=size) free(nodes);
    nodes = is_lo ? loNodes : hiNodes;
    N     = is_lo ? Nlo+Nrecv : Nhi+Nrecv;

    int offset = is_lo ? Nlo : Nhi;
    parallelNode_t *sendNodes = is_lo ? hiNodes : loNodes;

    //count how many entries from the halo buffer we're sending
    int NentriesSend=0;
    for (dlong n=0;n<Nsend;n++) {
      if (n==0 || abs(sendNodes[n].baseId)!=abs(sendNodes[n-1].baseId)) {
        NentriesSend++;
      }
    }
    levels[Nlevels].Nsend = NentriesSend;
    levels[Nlevels].sendIds = (dlong *) malloc(NentriesSend*sizeof(dlong));

    NentriesSend=0; //reset
    for (dlong n=0;n<Nsend;n++) {
      if (n==0 || abs(sendNodes[n].baseId)!=abs(sendNodes[n-1].baseId)) {
        levels[Nlevels].sendIds[NentriesSend++] = sendNodes[n].localId;
        // printf("rank %d, Send %d LocalId %d BaseId %d \n", rank, NentriesSend-1, sendNodes[n].localId, sendNodes[n].baseId);
      }
      sendNodes[n].localId = -1; //wipe the localId before sending
    }
    levels[Nlevels].o_sendIds = platform.malloc(NentriesSend*sizeof(dlong),
                                                levels[Nlevels].sendIds);

    //share the entry count with our partner
    MPI_Isend(&NentriesSend, 1, MPI_INT, partner, rank, comm, request+0);

    int NentriesRecv0=0, NentriesRecv1=0;
    if (Nmsg>0)
      MPI_Irecv(&NentriesRecv0, 1, MPI_INT, partner, partner, comm, request+1);
    if (Nmsg==2)
      MPI_Irecv(&NentriesRecv1, 1, MPI_INT, r_half-1, r_half-1, comm, request+2);

    MPI_Waitall(Nmsg+1, request, status);

    levels[Nlevels].Nrecv0 = NentriesRecv0;
    levels[Nlevels].Nrecv1 = NentriesRecv1;

    //send half the list to our partner
    MPI_Isend(sendNodes, Nsend,
              MPI_PARALLELNODE_T, partner, rank, comm, request+0);

    //recv new nodes from our partner(s)
    if (Nmsg>0)
      MPI_Irecv(nodes+offset,        Nrecv0,
                MPI_PARALLELNODE_T, partner, partner, comm, request+1);
    if (Nmsg==2)
      MPI_Irecv(nodes+offset+Nrecv0, Nrecv1,
                MPI_PARALLELNODE_T, r_half-1, r_half-1, comm, request+2);

    MPI_Waitall(Nmsg+1, request, status);

    free(sendNodes);

    //We now have a list of nodes who's destinations are in our half
    // of the hypercube
    //We now build the scatter into the haloBuffer

    //record the current order in newId
    for (dlong n=0;n<N;n++) nodes[n].newId = n;

    //sort the new node list by baseId to find matches
    std::sort(nodes, nodes+N,
            [](const parallelNode_t& a, const parallelNode_t& b) {
              if(abs(a.baseId) < abs(b.baseId)) return true; //group by abs(baseId)
              if(abs(a.baseId) > abs(b.baseId)) return false;

              return a.localId > b.localId; //positive localIds first
            });

    //fill localIds of new entries if possible, or give them an index
    dlong id = 0;
    for (dlong n=0;n<N;n++) {
      //for each baseId group
      if (n==0 || (abs(nodes[n].baseId)!=abs(nodes[n-1].baseId))) {
        id = nodes[n].localId; //get localId
        //no non-empty localId, must be a new node
        if (id==-1) id = NhaloExt++;
      }
      nodes[n].localId = id;
    }

    //sort back to first ordering
    std::sort(nodes, nodes+N,
            [](const parallelNode_t& a, const parallelNode_t& b) {
              return a.newId < b.newId;
            });

    if (Nmsg>0) {
      levels[Nlevels].recvIds0 = (dlong *) malloc(NentriesRecv0*sizeof(dlong));
      NentriesRecv0=0;
      for (dlong n=offset;n<offset+Nrecv0;n++) {
        if (n==offset || abs(nodes[n].baseId)!=abs(nodes[n-1].baseId)) {
          levels[Nlevels].recvIds0[NentriesRecv0++] = nodes[n].localId;
        }
      }
      levels[Nlevels].o_recvIds0 = platform.malloc(NentriesRecv0*sizeof(dlong),
                                                levels[Nlevels].recvIds0);
    }
    if (Nmsg==2) {
      levels[Nlevels].recvIds1 = (dlong *) malloc(NentriesRecv1*sizeof(dlong));
      NentriesRecv1=0;
      for (dlong n=offset+Nrecv0;n<N;n++) {
        if (n==offset+Nrecv0 || abs(nodes[n].baseId)!=abs(nodes[n-1].baseId)) {
          levels[Nlevels].recvIds1[NentriesRecv1++] = nodes[n].localId;
        }
      }
      levels[Nlevels].o_recvIds1 = platform.malloc(NentriesRecv1*sizeof(dlong),
                                                levels[Nlevels].recvIds1);
    }

    //sort the new node list by baseId
    std::sort(nodes, nodes+N,
            [](const parallelNode_t& a, const parallelNode_t& b) {
              if(abs(a.baseId) < abs(b.baseId)) return true; //group by abs(baseId)
              if(abs(a.baseId) > abs(b.baseId)) return false;

              return a.baseId < b.baseId; //positive ids first
            });

    //Shrink the size of the hypercube
    if (is_lo) {
      np = np_half;
    } else {
      np -= np_half;
      np_offset = r_half;
    }
    Nlevels++;
  }
  if (size>1) free(nodes);

  NsendMax=0, NrecvMax=0;
  for (int k=0;k<Nlevels;k++) {
    int Nsend = levels[k].Nsend;
    NsendMax = (Nsend>NsendMax) ? Nsend : NsendMax;
    int Nrecv = levels[k].Nrecv0 + levels[k].Nrecv1;
    NrecvMax = (Nrecv>NrecvMax) ? Nrecv : NrecvMax;
  }

  gatherHalo->setupRowBlocks(platform);
  scatterHalo = new ogsScatter_t(gatherHalo, platform);

  //make scratch space
  reallocOccaBuffer(sizeof(dfloat));
}

void ogsCrystalRouter_t::reallocOccaBuffer(size_t Nbytes) {
  if (o_haloBuf.size() < NhaloExt*Nbytes) {
    if (o_haloBuf.size()) o_haloBuf.free();
    haloBuf = platform.hostMalloc(NhaloExt*Nbytes,  nullptr, h_haloBuf);
    o_haloBuf = platform.malloc(NhaloExt*Nbytes);
  }
  if (o_sendBuf.size() < NsendMax*Nbytes) {
    if (o_sendBuf.size()) o_sendBuf.free();
    sendBuf = platform.hostMalloc(NsendMax*Nbytes,  nullptr, h_sendBuf);
    o_sendBuf = platform.malloc(NsendMax*Nbytes);
  }
  if (o_recvBuf.size() < NrecvMax*Nbytes) {
    if (o_recvBuf.size()) o_recvBuf.free();
    recvBuf = platform.hostMalloc(NrecvMax*Nbytes,  nullptr, h_recvBuf);
    o_recvBuf = platform.malloc(NrecvMax*Nbytes);
  }
}

ogsCrystalRouter_t::~ogsCrystalRouter_t() {
  // if(gatherHalo) gatherHalo->Free();
  // if(scatterHalo) scatterHalo->Free();

  if(levels) delete[] levels;

  if(o_haloBuf.size()) o_haloBuf.free();
  if(h_haloBuf.size()) h_haloBuf.free();
  if(o_sendBuf.size()) o_sendBuf.free();
  if(o_recvBuf.size()) o_recvBuf.free();
  if(h_sendBuf.size()) h_sendBuf.free();
  if(h_recvBuf.size()) h_recvBuf.free();
}

} //namespace ogs