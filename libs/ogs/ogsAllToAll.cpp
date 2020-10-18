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

void ogsAllToAll_t::Start(occa::memory& o_v, bool gpu_aware){

  const size_t Nbytes = sizeof(dfloat);
  reallocOccaBuffer(Nbytes);

  // assemble mpi send buffer by gathering halo nodes and scattering
  // them into the send buffer
  sendS->Apply(o_sendBuf, o_v);

  dlong Nsend = mpiSendOffsets[size];
  if (Nsend) {
    occa::device &device = platform.device;

    //wait for previous kernel to finish
    device.finish();

    if (!gpu_aware) {
      //switch streams to overlap data movement
      occa::stream currentStream = device.getStream();
      device.setStream(dataStream);

      o_sendBuf.copyTo(sendBuf, Nsend*Nbytes, 0, "async: true");

      device.setStream(currentStream);
    }
  }
}


void ogsAllToAll_t::Finish(occa::memory& o_v, bool gpu_aware){

  const size_t Nbytes = sizeof(dfloat);
  occa::device &device = platform.device;

  dlong Nsend = mpiRecvOffsets[size];
  if (Nsend) {
    //synchronize data stream to ensure the send buffer has arrived on host
    occa::stream currentStream = device.getStream();
    device.setStream(dataStream);
    device.finish();
    device.setStream(currentStream);
  }
  if (gpu_aware) {
    // collect everything needed with single MPI all to all
    MPI_Alltoallv(o_sendBuf.ptr(), mpiSendCounts, mpiSendOffsets, MPI_DFLOAT,
                  o_recvBuf.ptr(), mpiRecvCounts, mpiRecvOffsets, MPI_DFLOAT,
                  comm);
  } else {
    // collect everything needed with single MPI all to all
    MPI_Alltoallv(sendBuf, mpiSendCounts, mpiSendOffsets, MPI_DFLOAT,
                  recvBuf, mpiRecvCounts, mpiRecvOffsets, MPI_DFLOAT,
                  comm);
  }
  //if we recvieved anything via MPI, gather the recv buffer and scatter
  // it back to to original vector
  dlong Nrecv = mpiRecvOffsets[size];
  if (Nrecv) {
    if (!gpu_aware) {
      occa::stream currentStream = device.getStream();
      device.setStream(dataStream);

      // copy recv back to device
      o_recvBuf.copyFrom(recvBuf, Nrecv*Nbytes, 0, "async: true");

      device.finish();
      device.setStream(currentStream);
    }

    recvS->Apply(o_v, o_recvBuf);
  }
}

// compare on rank then local id
static int compareRank(const void *a, const void *b){

  parallelNode_t *fa = (parallelNode_t*) a;
  parallelNode_t *fb = (parallelNode_t*) b;

  if(fa->rank < fb->rank) return -1;
  if(fa->rank > fb->rank) return +1;

  if(fa->localId < fb->localId) return -1; //sort by local id
  if(fa->localId > fb->localId) return +1;

  return 0;
}

ogsAllToAll_t::ogsAllToAll_t(dlong recvN,
                             parallelNode_t* recvNodes,
                             dlong NgatherLocal,
                             ogsGather_t *gatherHalo,
                             dlong *indexMap,
                             MPI_Comm _comm,
                             platform_t &_platform):
  platform(_platform), comm(_comm) {

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  //make array of counters
  dlong *haloGatherCounts  = (dlong*) calloc(gatherHalo->Nrows,sizeof(dlong));

  dlong id=0;
  for (dlong n=0;n<recvN;n++) { //loop through nodes needed for gathering halo nodes
    if (n==0 || abs(recvNodes[n].baseId)!=abs(recvNodes[n-1].baseId)) { //for each baseId group
      //Find the node in this baseId group which was populated by this rank
      dlong origin=n;
      while (recvNodes[origin].rank!=rank) origin++;

      //map the baseId index to the coalesced ordering
      id = indexMap[recvNodes[origin].newId] - NgatherLocal;
    }
    recvNodes[n].localId = id; //record the coalesced index for this baseId

    haloGatherCounts[id]++;  //tally
  }

  // sort the list by rank to the order where they should recieved by MPI_Allgatherv
  qsort(recvNodes, recvN, sizeof(parallelNode_t), compareRank);

  //make mpi allgatherv counts and offsets
  mpiSendCounts = (int*) calloc(size, sizeof(int));
  mpiRecvCounts = (int*) calloc(size, sizeof(int));
  mpiSendOffsets = (int*) calloc(size+1, sizeof(int));
  mpiRecvOffsets = (int*) calloc(size+1, sizeof(int));

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

void ogsAllToAll_t::reallocOccaBuffer(size_t Nbytes) {
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

} //namespace ogs