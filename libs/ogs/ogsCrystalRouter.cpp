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

void ogsCrystalRouter_t::Start(occa::memory& o_v, bool gpu_aware){

  const size_t Nbytes = sizeof(dfloat);
  reallocOccaBuffer(Nbytes);

  // assemble mpi send buffer by gathering halo nodes
  gatherHalo->Apply(o_sBuf, o_v);

  if (gatherHalo->Nrows) {
    occa::device &device = platform.device;

    //wait for previous kernel to finish
    device.finish();

    if (!gpu_aware) {
      //switch streams to overlap data movement
      occa::stream currentStream = device.getStream();
      device.setStream(dataStream);

      o_sBuf.copyTo(sBuf, gatherHalo->Nrows*Nbytes, 0, "async: true");

      device.setStream(currentStream);
    }
  }
}


void ogsCrystalRouter_t::Finish(occa::memory& o_v, bool gpu_aware){

  const size_t Nbytes = sizeof(dfloat);
  occa::device &device = platform.device;

  if (gatherHalo->Nrows) {
    //synchronize data stream to ensure the send buffer has arrived on host
    occa::stream currentStream = device.getStream();
    device.setStream(dataStream);
    device.finish();
    device.setStream(currentStream);
  }

  if (gpu_aware){
    
    //post recvs
    for (int partner=0;partner<Npartners;partner++) {
      MPI_Irecv((o_sBuf+sOffsets[partner+1]*Nbytes).ptr(),
                sCounts[partner+1], MPI_DFLOAT,
                downstreamPartners[partner],
                downstreamPartners[partner],
                comm, requests+partner);
    }

    MPI_Waitall(Npartners, requests, statuses);

    occa::stream currentStream = device.getStream();
    device.setStream(dataStream);

    if (rank==0) {
      //apply a gather scatter on the root rank
      rootGS->Apply(o_sBuf);
      device.finish();
    } else {
      //partially gather nodes on this rank
      partialGather->Apply(o_gBuf, o_sBuf);
      device.finish();

      //send upstream
      MPI_Send(o_gBuf.ptr(), Nsend, MPI_DFLOAT, upstreamPartner, rank, comm);

      //recv gathered results
      MPI_Recv(o_gBuf.ptr(), Nsend, MPI_DFLOAT, upstreamPartner, rank, comm, MPI_STATUS_IGNORE);

      //scatter back to recieved ordering
      partialScatter->Apply(o_sBuf, o_gBuf);
      device.finish();
    }

    //post sends downstream
    for (int partner=0;partner<Npartners;partner++) {
      MPI_Isend((o_sBuf+sOffsets[partner+1]*Nbytes).ptr(),
                sCounts[partner+1], MPI_DFLOAT,
                downstreamPartners[partner],
                downstreamPartners[partner],
                comm, requests+partner);
    }
    MPI_Waitall(Npartners, requests, statuses);

    if (scatterHalo->Ncols) {
      scatterHalo->Apply(o_v, o_sBuf);
    }

    device.finish();
    device.setStream(currentStream);

  } else { // not gpu-aware
    //post recvs
    for (int partner=0;partner<Npartners;partner++) {
      MPI_Irecv((char*)sBuf+sOffsets[partner+1]*Nbytes,
                sCounts[partner+1], MPI_DFLOAT,
                downstreamPartners[partner],
                downstreamPartners[partner],
                comm, requests+partner);
    }
  
    MPI_Waitall(Npartners, requests, statuses);

    if (rank==0) {
      //apply a gather scatter on the root rank
      rootGS->Apply((dfloat*)sBuf);
    } else {
      //partially gather nodes on this rank
      partialGather->Apply((dfloat*)gBuf, (dfloat*)sBuf);

      //send upstream
      MPI_Send(gBuf, Nsend, MPI_DFLOAT, upstreamPartner, rank, comm);

      //recv gathered results
      MPI_Recv(gBuf, Nsend, MPI_DFLOAT, upstreamPartner, rank, comm, MPI_STATUS_IGNORE);

      //scatter back to recieved ordering
      partialScatter->Apply((dfloat*)sBuf, (dfloat*)gBuf);
    }

    //post sends downstream
    for (int partner=0;partner<Npartners;partner++) {
      MPI_Isend((char*)sBuf+sOffsets[partner+1]*Nbytes,
                sCounts[partner+1], MPI_DFLOAT,
                downstreamPartners[partner],
                downstreamPartners[partner],
                comm, requests+partner);
    }
    MPI_Waitall(Npartners, requests, statuses);
  
    // if we recieved anything via MPI, gather the recv buffer and scatter
    // it back to to original vector
    if (scatterHalo->Ncols) {
      
      occa::stream currentStream = device.getStream();
      device.setStream(dataStream);

      // copy recv back to device
      o_sBuf.copyFrom(sBuf, scatterHalo->Ncols*Nbytes, 0, "async: true");

      device.finish();
      device.setStream(currentStream);

      scatterHalo->Apply(o_v, o_sBuf);
    }
  }
}

// compare on baseId then rank then by localId
static int compareBaseId(const void *a, const void *b){

  parallelNode_t *fa = (parallelNode_t*) a;
  parallelNode_t *fb = (parallelNode_t*) b;

  if(abs(fa->baseId) < abs(fb->baseId)) return -1; //group by abs(baseId)
  if(abs(fa->baseId) > abs(fb->baseId)) return +1;

  if(fa->baseId > fb->baseId) return -1; //positive ids first
  if(fa->baseId < fb->baseId) return +1;

  if(fa->localId < fb->localId) return -1; //sort by local id
  if(fa->localId > fb->localId) return +1;

  return 0;
}

// compare on localId
static int compareLocalId(const void *a, const void *b){

  parallelNode_t *fa = (parallelNode_t*) a;
  parallelNode_t *fb = (parallelNode_t*) b;

  if(fa->localId < fb->localId) return -1;
  if(fa->localId > fb->localId) return +1;

  return 0;
}

/*
 *Crystal router performs the needed MPI communcation via a binary tree
 * traversal. The binary tree takes the following form:
 *                        0
 *                       / \
 *                      /   \
 *                     /     \
 *                    /       \
 *                   /         \
 *                  /           \
 *                 /             \
 *                /               \
 *               /                 \
 *              /                   \
 *             /                     \
 *            0                       1
 *           / \                     / \
 *          /   \                   /   \
 *         /     \                 /     \
 *        /       \               /       \
 *       /         \             /         \
 *      0           2           1           3
 *     / \         / \         / \         / \
 *    /   \       /   \       /   \       /   \
 *   0     4     2     6     1     5     3     7
 *  / \   / \   / \   / \   / \   / \   / \   / \
 * 0   8 4  12 2  10 6  14 1   9 5  13 3  11 7  15
 *
 *                    *   *   *
 *
 * For example, suppose a node is shared between ranks 5, 8, & 15.
 * The values at that node will begin in ranks 5, 8, & 15 and be sent up
 * the levels of the binary tree. When rank 1 recieves two of these values
 * from rank 5 and rank 3 (originating in rank 5 and rank 15, respectively)
 * rank 1 will perform a partial gather of the value before sending the
 * result to rank 0, which will have also recieved the third value from
 * rank 8. Rank 0 completes the gather, and the result is scattered back to
 * ranks 8, 5, and 15 (the latter 2 being sent via rank 1 then 5, and 1, 3,
 * 7, and finally 15, respectively)
 *
 * In general, each rank has a set of 'downstream' partners, i.e. the nodes
 * which connect to it from lower levels of the binary tree, and each rank,
 * except 0, has a single upstream partner, i.e. the rank to which it connects
 * to above it in the binary tree. To perform the Crystal Router exchange, each
 * rank receives a set of node values from each of its downstream partners. The
 * rank then gathers any repeated nodes and sets up a send buffer of node values
 * the are needed to send to thier upstream partner. The rank then waits to
 * recieve completely gathered nodes back from its upstream neighbor, scatters
 * the nodes recieved to send to each downstream partners, and sends those
 * values before scattering any local result back to the output array.
 */

static int LowestCommonAncestor(const int r1, const int r2) {
  //It is useful to know where a pair of nodes at ranks r1 and r2 will
  // intersect on their path up the binary tree

  //Since each node travels along the path [rank%2^{k}, rank%2^{k-1}, rank%2^{k-2},...]
  // in the tree, this amounts to finding the largest K such that
  // r1 == r2 (mod 2^K), at which point we know that the intersection is r = r1 % 2^K.

  if (r1==r2) return r1;

  int K=1;
  while (r1%(K<<1) == r2%(K<<1)) K <<=1;

  return (r1%K);
}

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

  gatherHalo = _gatherHalo;
  scatterHalo = new ogsScatter_t(gatherHalo, platform);

  //Determine our downstream partners and upstream partner
  // Our downstream partners are the ranks r + 2^k for
  // where r is this process's rank, and k=k0,k0+1,...
  // where k0 is smallest integer s.t. 2^k0 > r.
  // For example, for r=0 k0=1 and rank 0's downstream
  // partners are 2, 4, 8, 16, ..., and for r=3 k0=2 and
  // rank 3's downstream partners are 7, 11, 19, ....

  // K0 is smallest power of 2 that is > rank
  int K0=1;
  while (K0<=rank) K0<<=1;

  //upstream partner is rank - 2^(k0-1), e.g. 3's upstream is 1
  upstreamPartner=-1;
  if (rank>0) upstreamPartner = rank-(K0>>1);

  //count the number of downstream partners
  Npartners=0;
  int K=K0;
  while (rank+K<size) { Npartners++; K<<=1; }

  //record the downstream partners' ranks
  downstreamPartners = (int*) malloc(Npartners*sizeof(int));

  Npartners=0;
  K=K0;
  while (rank+K<size) {
    downstreamPartners[Npartners++] = rank+K;
    K<<=1;
  }

  requests = new MPI_Request[Npartners];
  statuses = new MPI_Status[Npartners];

  //count how many nodes we should be sending
  sCounts  = (int*) malloc((Npartners+1)*sizeof(int));
  sOffsets = (int*) malloc((Npartners+2)*sizeof(int));

  sCounts[0] = gatherHalo->Nrows; //first part of the buffer is the locally gathered halo nodes
  sOffsets[0] = 0;

  for (int partner=Npartners-1;partner>=0;partner--) {
    MPI_Recv(sCounts+partner+1, 1, MPI_INT,
             downstreamPartners[partner], 0,
             comm, statuses+partner);
  }
  for (int i=0;i<Npartners+1;i++) {
    sOffsets[i+1] = sOffsets[i] + sCounts[i];
  }

  //total number of nodes after scattering to send buffers
  sTotal = sOffsets[Npartners+1];

  //make a list of parallelNodes representing the values we
  // ultimately want
  parallelNode_t *sNodes = (parallelNode_t* )
                           malloc(sTotal*sizeof(parallelNode_t));

  //Note that half the ranks at the very bottom of the binary tree have
  // no downstream partners, and therefore recieved no extra nodes.
  //These ranks are the first to construct their lists

  for (dlong n=0;n<recvN;n++) { //loop through nodes needed for gathering halo nodes
    if (n==0 || abs(recvNodes[n].baseId)!=abs(recvNodes[n-1].baseId)) { //for each baseId group
      //Find the node in this baseId group which was orignally populated by this rank
      dlong origin=n;
      while (recvNodes[origin].rank!=rank) origin++;

      const dlong id = recvNodes[origin].newId; //get the group index
      const dlong sid = indexMap[id]-NgatherLocal;

      sNodes[sid] = recvNodes[n]; //copy a representative from this baseId group

      //loop through this baseId group and find at what rank the node should be fully gathered
      int k=1, destRank=recvNodes[n].rank;
      while (n+k<recvN && abs(recvNodes[n+k].baseId)==abs(recvNodes[n].baseId)) {
        destRank = LowestCommonAncestor(destRank,recvNodes[n+k].rank);
        k++;
      }
      sNodes[sid].destRank = destRank; //record the destination
    }
  }

  //Recieve node lists from downStream
  for (int partner=Npartners-1;partner>=0;partner--) {
    MPI_Recv(sNodes+sOffsets[partner+1],
             sCounts[partner+1], MPI_PARALLELNODE_T,
             downstreamPartners[partner], 0,
             comm, statuses+partner);
  }

  for (dlong n=0;n<sTotal;n++) {
    sNodes[n].localId = n; //reset local index
  }

  //we now have a list of nodes we should be sending to our downstream partners
  // build the gather to a compressed node list and build the list we should
  // send to our upstream

  // sort based on baseId (putting positive baseIds first) then by localId
  qsort(sNodes, sTotal, sizeof(parallelNode_t), compareBaseId);

  gTotal=0; //count how many unique nodes
  Nsend=0;  //count how many nodes we should recv from upstream
  for (dlong n=0;n<sTotal;n++) { //loop through nodes needed for gathering halo nodes
    if (n==0 || abs(sNodes[n].baseId)!=abs(sNodes[n-1].baseId)) { //for each baseId group
      gTotal++;

      //count if this baseId group needs to continue upstream
      if (sNodes[n].destRank<rank) Nsend++;
    }
    sNodes[n].newId=gTotal-1; //record the new ordering
  }

  //now that we know how many nodes we'll be sending, share that upstream
  if (upstreamPartner!=-1) {
    MPI_Send(&Nsend, 1, MPI_INT, upstreamPartner, 0, comm);
  }

  // sort the list back to local id ordering
  qsort(sNodes, sTotal, sizeof(parallelNode_t), compareLocalId);

  dlong *gIndexMap = (dlong*) malloc(gTotal*sizeof(dlong));
  for (dlong i=0;i<gTotal;i++) gIndexMap[i] = -1; //initialize map

  //make a list of nodes to send upstream
  parallelNode_t *gNodes = (parallelNode_t* )
                           malloc(Nsend*sizeof(parallelNode_t));

  partialGather = new ogsGather_t();

  partialGather->Nrows = gTotal;
  partialGather->Ncols = sTotal;

  int *gatherCounts  = (int*) calloc(partialGather->Nrows,sizeof(int));
  dlong cnt=0;
  dlong cnt2=Nsend; //indexing for nodes which stop at this rank
  for (dlong i=0;i<sTotal;i++) {
    dlong newId = sNodes[i].newId; //get the new baseId group id

    //record a new index if we've not encoutered this baseId group before
    if (gIndexMap[newId]==-1) {
      if (sNodes[i].destRank<rank) {
        gIndexMap[newId] = cnt;
        gNodes[cnt] = sNodes[i]; //copy the nodes here to send upstream
        cnt++;
      } else {
        gIndexMap[newId] = cnt2++;
      }
    }

    const dlong gid = gIndexMap[newId];
    sNodes[i].newId = gid; //reorder
    gatherCounts[gid]++;  //tally
  }
  free(gIndexMap);

  //send list of nodes upstream
  if (upstreamPartner!=-1) {
    MPI_Send(gNodes, Nsend, MPI_PARALLELNODE_T, upstreamPartner, 0, comm);
  }

  //make local row offsets
  partialGather->rowStarts = (dlong*) calloc(partialGather->Nrows+1,sizeof(dlong));
  for (dlong i=0;i<partialGather->Nrows;i++) {
    partialGather->rowStarts[i+1] = partialGather->rowStarts[i] + gatherCounts[i];

    //reset counters
    gatherCounts[i] = 0;
  }
  partialGather->nnz = partialGather->rowStarts[partialGather->Nrows];

  partialGather->colIds = (dlong*) calloc(partialGather->nnz+1,sizeof(dlong)); //extra entry so the occa buffer will actually exist

  for (dlong i=0;i<sTotal;i++) {
    const dlong gid = sNodes[i].newId;

    const dlong soffset = partialGather->rowStarts[gid];
    const int sindex  = gatherCounts[gid];
    partialGather->colIds[soffset+sindex] = sNodes[i].localId; //record id
    gatherCounts[gid]++;
  }
  free(sNodes); //done with these
  free(gatherCounts);

  partialGather->o_rowStarts = platform.malloc((partialGather->Nrows+1)*sizeof(dlong), partialGather->rowStarts);
  partialGather->o_colIds = platform.malloc((partialGather->nnz+1)*sizeof(dlong), partialGather->colIds);

  if (rank==0) {
    //root rank makes a gatherScatter operator
    rootGS = new ogsGatherScatter_t();
    rootGS->Nrows = partialGather->Nrows;
    rootGS->gather  = partialGather;
    rootGS->scatter = partialGather;
    rootGS->setupRowBlocks(platform);
  } else {
    //other ranks setup a gather operator
    partialGather->setupRowBlocks(platform);
    partialScatter = new ogsScatter_t(partialGather, platform);
  }

  gatherHalo->setupRowBlocks(platform);

  //free up the node lists
  MPI_Barrier(comm);
  free(gNodes);

  //make scratch space
  reallocOccaBuffer(sizeof(dfloat));
}

void ogsCrystalRouter_t::reallocOccaBuffer(size_t Nbytes) {
  if (o_sBuf.size() < sTotal*Nbytes) {
    if (o_sBuf.size()) o_sBuf.free();
    sBuf = platform.hostMalloc(sTotal*Nbytes,  nullptr, h_sBuf);
    o_sBuf = platform.malloc(sTotal*Nbytes);
  }
  if (o_gBuf.size() < gTotal*Nbytes) {
    if (o_gBuf.size()) o_gBuf.free();
    gBuf = platform.hostMalloc(gTotal*Nbytes,  nullptr, h_gBuf);
    o_gBuf = platform.malloc(gTotal*Nbytes);
  }
}

} //namespace ogs