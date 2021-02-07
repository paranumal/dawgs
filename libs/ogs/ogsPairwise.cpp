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

void ogsPairwise_t::Start(occa::memory& o_v, bool gpu_aware, bool overlap){

  const size_t Nbytes = sizeof(dfloat);
  reallocOccaBuffer(Nbytes);

  occa::device &device = platform.device;

  //get current stream
  occa::stream currentStream = device.getStream();

  dlong Nsend = sendOffsets[NranksSend];
  if (Nsend) {
    //if overlapping the halo kernels, switch streams
    if (overlap)
      device.setStream(dataStream);

    // assemble mpi send buffer by gathering halo nodes and scattering
    // them into the send buffer
    sendS->Apply(o_sendBuf, o_v);

    //if not overlapping, wait for kernel to finish on default stream
    if (!overlap)
      device.finish();

    //if using gpu-aware mpi, queue the data movement into dataStream
    if (!gpu_aware) {
      device.setStream(dataStream);
      o_sendBuf.copyTo(sendBuf, Nsend*Nbytes, 0, "async: true");
    }

    device.setStream(currentStream);
  }
}


void ogsPairwise_t::Finish(occa::memory& o_v, bool gpu_aware, bool overlap){

  const size_t Nbytes = sizeof(dfloat);
  occa::device &device = platform.device;

  //get current stream
  occa::stream currentStream = device.getStream();

  dlong Nsend = sendOffsets[NranksSend];
  if (Nsend) {
    //synchronize data stream to ensure the send buffer is ready to send
    device.setStream(dataStream);
    device.finish();
  }

  char *sendPtr, *recvPtr;
  if (gpu_aware) { //device pointer
    sendPtr = (char*)o_sendBuf.ptr();
    recvPtr = (char*)o_recvBuf.ptr();
  } else { //host pointer
    sendPtr = (char*)sendBuf;
    recvPtr = (char*)recvBuf;
  }

  //post recvs
  for (int r=0;r<NranksRecv;r++) {
    MPI_Irecv(recvPtr+recvOffsets[r]*Nbytes,
              recvCounts[r], MPI_DFLOAT, recvRanks[r],
              recvRanks[r], comm, requests+r);
  }

  //post sends
  for (int r=0;r<NranksSend;r++) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(overhead));
    MPI_Isend(sendPtr+sendOffsets[r]*Nbytes,
              sendCounts[r], MPI_DFLOAT, sendRanks[r],
              rank, comm, requests+NranksRecv+r);
  }
  MPI_Waitall(NranksRecv+NranksSend, requests, statuses);

  //if we recvieved anything via MPI, gather the recv buffer and scatter
  // it back to to original vector
  dlong Nrecv = recvOffsets[NranksRecv];
  if (Nrecv) {
    if (!gpu_aware) {
      // copy recv back to device
      device.setStream(dataStream);
      o_recvBuf.copyFrom(recvBuf, Nrecv*Nbytes, 0, "async: true");
      if (!overlap) device.finish(); //wait for transfer to finish if not overlapping halo kernel
    }

    //if overlapping the halo kernels, switch streams
    if (overlap) {
      device.setStream(dataStream);
    } else {
      device.setStream(currentStream);
    }

    recvS->Apply(o_v, o_recvBuf);

    if (overlap) { //if overlapping halo kernels wait for kernel to finish
      device.finish();
    }
  }

  device.setStream(currentStream);
}

ogsPairwise_t::ogsPairwise_t(dlong recvN,
                             parallelNode_t* recvNodes,
                             dlong NgatherLocal,
                             ogsGather_t *gatherHalo,
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

  //make array of counters
  dlong *haloGatherCounts  = (dlong*) calloc(gatherHalo->Nrows,sizeof(dlong));

  for (dlong n=0;n<recvN;n++) { //loop through nodes needed for gathering halo nodes
    dlong id = recvNodes[n].localId; //coalesced index for this baseId
    haloGatherCounts[id]++;  //tally
  }

  // sort the list by rank to the order where they should recieved by MPI_Allgatherv
  std::sort(recvNodes, recvNodes+recvN,
            [](const parallelNode_t& a, const parallelNode_t& b) {
              if(a.rank < b.rank) return true; //group by rank
              if(a.rank > b.rank) return false;

              return a.localId < b.localId; //then order by localId
            });

  //make mpi allgatherv counts and offsets
  int *mpiSendCounts = (int*) calloc(size, sizeof(int));
  int *mpiRecvCounts = (int*) calloc(size, sizeof(int));
  int *mpiSendOffsets = (int*) calloc(size+1, sizeof(int));
  int *mpiRecvOffsets = (int*) calloc(size+1, sizeof(int));

  //make ops for gathering halo nodes after an MPI_Allgatherv
  postmpi = new ogsGather_t();

  postmpi->Nrows = gatherHalo->Nrows;
  postmpi->rowStarts = (dlong*) calloc(postmpi->Nrows+1,sizeof(dlong));
  for (dlong i=0;i<postmpi->Nrows;i++) {
    postmpi->rowStarts[i+1] = postmpi->rowStarts[i] + haloGatherCounts[i];
    haloGatherCounts[i] = 0;
  }
  postmpi->nnz = postmpi->rowStarts[postmpi->Nrows];
  postmpi->colIds = (dlong*) calloc(postmpi->nnz+1,sizeof(dlong));

  dlong *recvIds = (dlong*) calloc(postmpi->nnz,sizeof(dlong));

  for (dlong n=0;n<recvN;n++) {
    //count what rank this is recieved from
    mpiRecvCounts[recvNodes[n].rank]++;

    recvIds[n] = recvNodes[n].newId;

    const dlong hid = recvNodes[n].localId;
    const dlong soffset = postmpi->rowStarts[hid];
    const int sindex  = haloGatherCounts[hid];
    postmpi->colIds[soffset+sindex] = n; //record id
    haloGatherCounts[hid]++;
  }


  //shared counts
  MPI_Alltoall(mpiRecvCounts, 1, MPI_INT,
               mpiSendCounts, 1, MPI_INT, comm);

  //cumulative sum
  for (int r=0;r<size;r++) {
    mpiSendOffsets[r+1] = mpiSendOffsets[r]+mpiSendCounts[r];
    mpiRecvOffsets[r+1] = mpiRecvOffsets[r]+mpiRecvCounts[r];
  }

  dlong sendN = mpiSendOffsets[size];
  dlong *sendIds = (dlong*) malloc(sendN*sizeof(dlong));

  //Share the list of newIds we expect to recieve from each rank
  MPI_Alltoallv(recvIds, mpiRecvCounts, mpiRecvOffsets, MPI_DLONG,
                sendIds, mpiSendCounts, mpiSendOffsets, MPI_DLONG,
                comm);

  //free up the send space
  MPI_Barrier(comm);
  free(recvIds);

  //compress the send/recv counts to pairwise exchanges
  NranksSend=0;
  NranksRecv=0;
  for (int r=0;r<size;r++) {
    NranksSend += (mpiSendCounts[r]>0) ? 1 : 0;
    NranksRecv += (mpiRecvCounts[r]>0) ? 1 : 0;
  }

  sendRanks   = (int*) calloc(NranksSend, sizeof(int));
  recvRanks   = (int*) calloc(NranksRecv, sizeof(int));
  sendCounts  = (int*) calloc(NranksSend, sizeof(int));
  recvCounts  = (int*) calloc(NranksRecv, sizeof(int));
  sendOffsets = (int*) calloc(NranksSend+1, sizeof(int));
  recvOffsets = (int*) calloc(NranksRecv+1, sizeof(int));

  //reset
  NranksSend=0;
  NranksRecv=0;
  for (int r=0;r<size;r++) {
    if (mpiSendCounts[r]>0) {
      sendRanks[NranksSend]  = r;
      sendCounts[NranksSend] = mpiSendCounts[r];
      sendOffsets[NranksSend] = mpiSendOffsets[r];
      NranksSend++;
    }
    if (mpiRecvCounts[r]>0) {
      recvRanks[NranksRecv]   = r;
      recvCounts[NranksRecv]  = mpiRecvCounts[r];
      recvOffsets[NranksRecv] = mpiRecvOffsets[r];
      NranksRecv++;
    }
  }
  sendOffsets[NranksSend] = mpiSendOffsets[size];
  recvOffsets[NranksRecv] = mpiRecvOffsets[size];

  requests = new MPI_Request[NranksSend+NranksRecv];
  statuses = new MPI_Status[NranksSend+NranksRecv];

  free(mpiSendCounts);
  free(mpiRecvCounts);
  free(mpiSendOffsets);
  free(mpiRecvOffsets);

  //reset counters
  for (dlong i=0;i<gatherHalo->Nrows;i++) haloGatherCounts[i] = 0;

  for (dlong n=0;n<sendN;n++) { //loop through nodes we need to send
    const dlong hid = indexMap[sendIds[n]] - NgatherLocal;
    haloGatherCounts[hid]++;  //tally
  }

  //make ops for scattering halo nodes before sending via MPI_Allgatherv
  prempi = new ogsGather_t();
  prempi->Nrows = gatherHalo->Nrows;
  prempi->rowStarts       = (dlong*) calloc(prempi->Nrows+1,sizeof(dlong));
  for (dlong i=0;i<prempi->Nrows;i++) {
    prempi->rowStarts[i+1] = prempi->rowStarts[i] + haloGatherCounts[i];
    haloGatherCounts[i] = 0;
  }
  prempi->nnz = prempi->rowStarts[prempi->Nrows];

  prempi->colIds = (dlong*) calloc(prempi->nnz+1,sizeof(dlong));

  for (dlong n=0;n<sendN;n++) { //loop through nodes we need to send
    const dlong hid = indexMap[sendIds[n]] - NgatherLocal;
    const dlong soffset = prempi->rowStarts[hid];
    const int sindex  = haloGatherCounts[hid];
    prempi->colIds[soffset+sindex] = n; //record id
    haloGatherCounts[hid]++;
  }

  free(haloGatherCounts);
  free(sendIds);

  prempi->o_rowStarts = platform.malloc((prempi->Nrows+1)*sizeof(dlong), prempi->rowStarts);
  prempi->o_colIds  = platform.malloc((prempi->nnz+1)*sizeof(dlong), prempi->colIds);

  postmpi->o_rowStarts  = platform.malloc((postmpi->Nrows+1)*sizeof(dlong), postmpi->rowStarts);
  postmpi->o_colIds  = platform.malloc((postmpi->nnz+1)*sizeof(dlong), postmpi->colIds);

  //make gatherScatter operator
  sendS = new ogsGatherScatter_t();
  sendS->Nrows = gatherHalo->Nrows;
  sendS->gather  = gatherHalo;
  sendS->scatter = prempi;

  recvS = new ogsGatherScatter_t();
  recvS->Nrows = gatherHalo->Nrows;
  recvS->gather  = postmpi;
  recvS->scatter = gatherHalo;

  //divide the list of colIds into roughly equal sized blocks so that each
  // threadblock loads approximately an equal amount of data
  sendS->setupRowBlocks(platform);
  recvS->setupRowBlocks(platform);

  //make scratch space
  reallocOccaBuffer(sizeof(dfloat));
}

void ogsPairwise_t::reallocOccaBuffer(size_t Nbytes) {
  if (o_sendBuf.size() < prempi->nnz*Nbytes) {
    if (o_sendBuf.size()) o_sendBuf.free();
    sendBuf = platform.hostMalloc(prempi->nnz*Nbytes,  nullptr, h_sendBuf);
    o_sendBuf = platform.malloc(prempi->nnz*Nbytes);
  }
  if (o_recvBuf.size() < postmpi->nnz*Nbytes) {
    if (o_recvBuf.size()) o_recvBuf.free();
    recvBuf = platform.hostMalloc(postmpi->nnz*Nbytes,  nullptr, h_recvBuf);
    o_recvBuf = platform.malloc(postmpi->nnz*Nbytes);
  }
}

ogsPairwise_t::~ogsPairwise_t() {
  if(prempi) prempi->Free();
  if(postmpi) postmpi->Free();

  if(sendS) sendS->Free();
  if(recvS) recvS->Free();

  if(o_sendBuf.size()) o_sendBuf.free();
  if(o_recvBuf.size()) o_recvBuf.free();
  if(h_sendBuf.size()) h_sendBuf.free();
  if(h_recvBuf.size()) h_recvBuf.free();

  if(sendRanks) free(sendRanks);
  if(recvRanks) free(recvRanks);
  if(sendCounts) free(sendCounts);
  if(recvCounts) free(recvCounts);
  if(sendOffsets) free(sendOffsets);
  if(recvOffsets) free(recvOffsets);
  if(requests) free(requests);
  if(statuses) free(statuses);
}

} //namespace ogs