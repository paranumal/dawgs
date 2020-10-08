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

// compare on destRank
static int compareDestRank(const void *a, const void *b){

  parallelNode_t *fa = (parallelNode_t*) a;
  parallelNode_t *fb = (parallelNode_t*) b;

  if(fa->destRank < fb->destRank) return -1;
  if(fa->destRank > fb->destRank) return +1;

  return 0;
}

// compare on baseId then rank then by localId
static int compareBaseId(const void *a, const void *b){

  parallelNode_t *fa = (parallelNode_t*) a;
  parallelNode_t *fb = (parallelNode_t*) b;

  if(abs(fa->baseId) < abs(fb->baseId)) return -1; //group by abs(baseId)
  if(abs(fa->baseId) > abs(fb->baseId)) return +1;

  if(fa->baseId > fb->baseId) return -1; //positive ids first
  if(fa->baseId < fb->baseId) return +1;

  if(fa->rank < fb->rank) return -1; //sort by rank
  if(fa->rank > fb->rank) return +1;

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

void ogs_t::Setup(dlong _N, hlong *ids, MPI_Comm _comm, int verbose){

  //release resources if this ogs was setup before
  Free();

  N = _N;
  comm = _comm;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  //count how many ids are non-zero
  dlong Nids=0;
  for (dlong n=0;n<N;n++)
    if (ids[n]!=0) Nids++;

  // make list of nodes
  parallelNode_t *nodes = (parallelNode_t* )
                               malloc(Nids*sizeof(parallelNode_t));

  //fill the data (squeezing out zero ids)
  Nids=0;
  for (dlong n=0;n<N;n++) {
    if (ids[n]!=0) {
      nodes[Nids].localId = n; //record original id
      nodes[Nids].baseId = ids[n]; //record global id
      nodes[Nids].rank = rank;
      nodes[Nids].destRank = abs(ids[n]) % size;
      Nids++;
    }
  }

  // sort based on baseId (putting positive baseIds first), then localId
  qsort(nodes, Nids, sizeof(parallelNode_t), compareBaseId);

  //count how many unique global Ids we have on this rank, and flag
  // baseId groups as either 1 or -1 based on whether there is a
  // positive baseId present.
  dlong NbaseIds=0;
  for (dlong n=0;n<Nids;n++) {
    if (n==0 || abs(nodes[n].baseId)!=abs(nodes[n-1].baseId)) {
      // record whether there is a non-negative id on this rank
      nodes[n].sign = (nodes[n].baseId>0) ? 1 : -1;
      NbaseIds++;
    } else {
      nodes[n].sign = nodes[n-1].sign;
    }
    nodes[n].newId=NbaseIds-1; //record the new ordering
  }

  //make a list of offsets so we can find baseId groups easily
  dlong *baseIdOffsets = (dlong*) calloc(NbaseIds+1,sizeof(dlong));

  //count the size of each gathered globalId
  NbaseIds=0;
  for (dlong n=0;n<Nids;n++) {
    if (n==0 || abs(nodes[n].baseId)!=abs(nodes[n-1].baseId)) {
      NbaseIds++;
    }
    baseIdOffsets[NbaseIds]++;
  }

  //cumulative sum
  for (dlong n=1;n<=NbaseIds;n++)
    baseIdOffsets[n]+=baseIdOffsets[n-1];

  // make list of parallel nodes with unique global Ids for communication
  parallelNode_t *sendNodes = (parallelNode_t* )
                                   malloc(NbaseIds*sizeof(parallelNode_t));

  //fill the data (copy the first node of each baseId group)
  for (dlong n=0;n<NbaseIds;n++) {
    sendNodes[n] = nodes[baseIdOffsets[n]];
  }

  // sort based on destination rank
  qsort(sendNodes, NbaseIds, sizeof(parallelNode_t), compareDestRank);

  //count number of ids we're sending
  int *sendCounts = (int*) calloc(size, sizeof(int));
  int *recvCounts = (int*) calloc(size, sizeof(int));
  int *sendOffsets = (int*) calloc(size+1, sizeof(int));
  int *recvOffsets = (int*) calloc(size+1, sizeof(int));

  for (dlong n=0;n<NbaseIds;n++) {
    sendCounts[sendNodes[n].destRank]++;
  }

  MPI_Alltoall(sendCounts, 1, MPI_INT,
               recvCounts, 1, MPI_INT, comm);

  for (int r=0;r<size;r++) {
    sendOffsets[r+1] = sendOffsets[r]+sendCounts[r];
    recvOffsets[r+1] = recvOffsets[r]+recvCounts[r];
  }
  dlong recvN = recvOffsets[size]; //total ids to recv

  parallelNode_t *recvNodes = (parallelNode_t* )
                                   malloc(recvN*sizeof(parallelNode_t));

  //Send all the nodes to their destination rank.
  MPI_Alltoallv(sendNodes, sendCounts, sendOffsets, ogs::MPI_PARALLELNODE_T,
                recvNodes, recvCounts, recvOffsets, ogs::MPI_PARALLELNODE_T,
                comm);

  MPI_Barrier(comm);
  free(sendNodes);

  // sort based on base ids (putting positive ids first) then rank, then local id
  qsort(recvNodes, recvN, sizeof(parallelNode_t), compareBaseId);

  // We now have a collection of nodes associated with some subset of all global Ids
  // Our list is sorted by baseId to group nodes with the same globalId together
  // We now want to flag which nodes are shared via MPI

  //count how many unique baseIds we have on this rank
  dlong NrecvBaseIds=0;
  for (dlong n=0;n<recvN;n++) {
    if (n==0 || abs(recvNodes[n].baseId)!=abs(recvNodes[n-1].baseId))
      NrecvBaseIds++;
  }

  dlong *recvBaseIdOffsets = (dlong*) calloc(NrecvBaseIds+1,sizeof(dlong));

  //count the size of each gathered globalId
  NrecvBaseIds=0;
  for (dlong n=0;n<recvN;n++) {
    if (n==0 || abs(recvNodes[n].baseId)!=abs(recvNodes[n-1].baseId)) {
      NrecvBaseIds++;
    }
    recvBaseIdOffsets[NrecvBaseIds]++;
  }

  //cumulative sum
  for (dlong n=1;n<=NrecvBaseIds;n++)
    recvBaseIdOffsets[n]+=recvBaseIdOffsets[n-1];

  //mark the nodes that are from multiple ranks
  for (dlong n=0;n<NrecvBaseIds;n++) { //for each gathered baseId
    const dlong start = recvBaseIdOffsets[n];
    const dlong end   = recvBaseIdOffsets[n+1];

    //if there is more than one entry for this baseId, they
    // must have come from different ranks
    if (end-start>1) {
      for (dlong i=start;i<end;i++) {
        recvNodes[i].sign *= 2; //flag node as a halo node
      }
    }
  }

  //at this point each collection of baseIds either has a single node
  // with sign = +-1, meaning all the nodes with this baseId are on the
  // same rank, or have sign = +-2, meaning that baseId must be communicated

  // Each rank has a set of shared global Ids and for each global id, that
  // rank knows what MPI ranks participate in gathering. We now send this
  // information to the involved ranks.

  //reset sendCounts
  for (int r=0;r<size;r++) sendCounts[r]=0;

  //count how many nodes we're sending
  for (dlong n=0;n<NrecvBaseIds;n++) { //for each gathered baseId
    const dlong start = recvBaseIdOffsets[n];
    const dlong end   = recvBaseIdOffsets[n+1];

    if (end-start>1) {
      for (dlong i=start;i<end;i++) {
        sendCounts[recvNodes[i].rank] += end-start;
      }
    }
  }

  //share counts
  MPI_Alltoall(sendCounts, 1, MPI_INT,
               recvCounts, 1, MPI_INT, comm);

  //cumulative sum
  for (int r=0;r<size;r++) {
    sendOffsets[r+1] = sendOffsets[r]+sendCounts[r];
    recvOffsets[r+1] = recvOffsets[r]+recvCounts[r];
  }

  //remake a send buffer
  sendNodes = (parallelNode_t* )
              malloc(sendOffsets[size]*sizeof(parallelNode_t));

  //reset sendCounts
  for (int r=0;r<size;r++) sendCounts[r]=0;

  for (dlong n=0;n<NrecvBaseIds;n++) { //for each gathered baseId
    const dlong start = recvBaseIdOffsets[n];
    const dlong end   = recvBaseIdOffsets[n+1];

    if (end-start>1) {
      //for every rank participating in gathering this baseId
      for (dlong i=start;i<end;i++) {
        dlong cnt =   sendOffsets[recvNodes[i].rank]
                    + sendCounts[recvNodes[i].rank];

        for (dlong j=start;j<end;j++) { //write the full gather data into send buffer
          sendNodes[cnt++] = recvNodes[j];
        }
        sendCounts[recvNodes[i].rank] += end-start;
      }
    }
  }
  free(recvBaseIdOffsets);

  //remake the recv buffer (we don't need the old one anymore)
  free(recvNodes);
  recvN = recvOffsets[size];
  recvNodes = (parallelNode_t* )
              malloc(recvOffsets[size]*sizeof(parallelNode_t));

  //Share all the gathering info
  MPI_Alltoallv(sendNodes, sendCounts, sendOffsets, ogs::MPI_PARALLELNODE_T,
                recvNodes, recvCounts, recvOffsets, ogs::MPI_PARALLELNODE_T,
                comm);

  //free up the send space
  MPI_Barrier(comm);
  free(sendNodes);
  free(sendCounts);
  free(recvCounts);
  free(sendOffsets);
  free(recvOffsets);

  // sort based on baseId
  qsort(recvNodes, recvN, sizeof(parallelNode_t), compareBaseId);

  // We now have a list of parallelNodes which have been flagged as shared.
  // For each node, we also have a list of what ranks that node
  // is shared with

  for (dlong n=0;n<recvN;n++) { //loop through nodes needed for gathering halo nodes
    if (n==0 || abs(recvNodes[n].baseId)!=abs(recvNodes[n-1].baseId)) { //for each baseId group
      //Find the node in this baseId group which was orignally populated by this rank
      dlong origin=n;
      while (recvNodes[origin].rank!=rank) origin++;

      dlong id = recvNodes[origin].newId; //get the group index

      //for each local node in this baseId group
      const dlong start = baseIdOffsets[id];
      const dlong end   = baseIdOffsets[id+1];
      for (dlong i=start;i<end;i++) {
        nodes[i].sign *= 2; //flag as a halo node
      }
    }
  }
  free(baseIdOffsets);

  // We now know which of our local list of nodes are halo nodes

  //count some things
  //gatherLocal->Nrows counts the number of global ids local to
  // this rank
  //localGather.Nrows counts the number of global ids local to
  // this rank with at least one positive scattered node
  //gatherHalo->Nrows counts the number of global ids shared via
  // MPI
  //haloGather.Nrows counts the number of global ids shared via
  // MPI, with at least one positive scattered node on this rank
  gatherLocal = new ogsGather_t();
  gatherHalo  = new ogsGather_t();

  Nlocal=0; Nhalo=0;
  for (dlong n=0;n<Nids;n++) {
    if (n==0 || abs(nodes[n].baseId)!=abs(nodes[n-1].baseId)) {
      if      (abs(nodes[n].sign)==1) gatherLocal->Nrows++;
      else if (abs(nodes[n].sign)==2) gatherHalo->Nrows++;
    }
  }

  //We should have NbaseIds ==  gatherLocal->Nrows
  //                           +gatherHalo->Nrows

  // sort the list back to local id ordering
  qsort(nodes, Nids, sizeof(parallelNode_t), compareLocalId);

  // When sorted by baseId, we numbered the baseId groups with a new index,
  // newId. We use this index to reorder the baseId groups based on
  // the order we encouter them in their original ordering.
  dlong *indexMap = (dlong*) malloc(NbaseIds*sizeof(dlong));
  for (dlong i=0;i<NbaseIds;i++) indexMap[i] = -1; //initialize map

  //tally up how many nodes are being gathered to each gatherNode and
  //  map to a local ordering
  dlong *localGatherCounts = (dlong*) calloc(gatherLocal->Nrows,sizeof(dlong));
  dlong *haloGatherCounts  = (dlong*) calloc(gatherHalo->Nrows,sizeof(dlong));

  dlong localGatherCnt = 0;                                     //start point for positive local gather nodes
  dlong haloGatherCnt   = gatherLocal->Nrows;              //start point for positive halo gather nodes
  for (dlong i=0;i<Nids;i++) {
    dlong newId = nodes[i].newId; //get the new baseId group id

    //record a new index if we've not encoutered this baseId group before
    if (indexMap[newId]==-1) {
      if      (abs(nodes[i].sign)== 1) indexMap[newId] = localGatherCnt++;
      else if (abs(nodes[i].sign)== 2) indexMap[newId] = haloGatherCnt++;
    }

    const dlong gid = indexMap[newId];
    nodes[i].newId = gid; //reorder

    if (abs(nodes[i].sign)== 1) { //local
      localGatherCounts[gid]++;  //tally
    } else if (abs(nodes[i].sign)== 2) { //halo
      const dlong hid = gid - gatherLocal->Nrows;
      haloGatherCounts[hid]++;  //tally
    }
  }


  //make local row offsets
  gatherLocal->rowStarts = (dlong*) calloc(gatherLocal->Nrows+1,sizeof(dlong));
  for (dlong i=0;i<gatherLocal->Nrows;i++) {
    gatherLocal->rowStarts[i+1] = gatherLocal->rowStarts[i] + localGatherCounts[i];

    //reset counters
    localGatherCounts[i] = 0;
  }
  gatherLocal->nnz = gatherLocal->rowStarts[gatherLocal->Nrows];

  gatherLocal->colIds = (dlong*) calloc(gatherLocal->nnz+1,sizeof(dlong)); //extra entry so the occa buffer will actually exist

  //make halo row offsets
  gatherHalo->rowStarts = (dlong*) calloc(gatherHalo->Nrows+1,sizeof(dlong));
  for (dlong i=0;i<gatherHalo->Nrows;i++) {
    gatherHalo->rowStarts[i+1] = gatherHalo->rowStarts[i] + haloGatherCounts[i];
    haloGatherCounts[i] = 0;
  }
  gatherHalo->nnz = gatherHalo->rowStarts[gatherHalo->Nrows];

  gatherHalo->colIds = (dlong*) calloc(gatherHalo->nnz+1,sizeof(dlong));

  for (dlong i=0;i<Nids;i++) {
    const dlong gid = nodes[i].newId;

    if (gid<gatherLocal->Nrows) { //local gather group
      const dlong soffset = gatherLocal->rowStarts[gid];
      const int sindex  = localGatherCounts[gid];
      gatherLocal->colIds[soffset+sindex] = nodes[i].localId; //record id
      localGatherCounts[gid]++;
    } else {
      const dlong hid = gid - gatherLocal->Nrows;
      const dlong soffset = gatherHalo->rowStarts[hid];
      const int sindex  = haloGatherCounts[hid];
      gatherHalo->colIds[soffset+sindex] = nodes[i].localId; //record id
      haloGatherCounts[hid]++;
    }
  }
  free(localGatherCounts);

  //with that, we're done with the local nodes list
  free(nodes);

  gatherLocal->o_rowStarts = platform.malloc((gatherLocal->Nrows+1)*sizeof(dlong), gatherLocal->rowStarts);
  gatherLocal->o_colIds = platform.malloc((gatherLocal->nnz+1)*sizeof(dlong), gatherLocal->colIds);

  gatherHalo->o_rowStarts = platform.malloc((gatherHalo->Nrows+1)*sizeof(dlong), gatherHalo->rowStarts);
  gatherHalo->o_colIds = platform.malloc((gatherHalo->nnz+1)*sizeof(dlong), gatherHalo->colIds);

  //make gatherScatter operator
  gsLocalS = new ogsGatherScatter_t();
  gsLocalS->Nrows = gatherLocal->Nrows;
  gsLocalS->gather  = gatherLocal;
  gsLocalS->scatter = gatherLocal;

  //divide the list of colIds into roughly equal sized blocks so that each
  // threadblock loads approximately an equal amount of data
  gsLocalS->setupRowBlocks(platform);

  Nlocal = gatherLocal->Nrows;
  Nhalo = gatherHalo->Nrows;

  //total number of owned gathered nodes
  Ngather = Nlocal+Nhalo;

  hlong NgatherLocal = (hlong) Ngather;
  MPI_Allreduce(&NgatherLocal, &(NgatherGlobal), 1, MPI_HLONG, MPI_SUM, comm);

  // At this point, we've setup gs operators to gather/scatter the purely local nodes,
  // and gather/scatter the shared halo nodes to/from a coalesced ordering. We now
  // need gs operators to scatter/gather the coalesced halo nodes to/from the expected
  // orderings for MPI communications.
  exchange = new ogsAllToAll_t(recvN, recvNodes, Nlocal,
                               gatherHalo, indexMap, comm, platform);

  //we're now done with the recvNodes list
  free(recvNodes);
  free(indexMap);
}

ogs_t::ogs_t(platform_t& _platform): platform(_platform) {
  //Keep track of how many gs handles we've created, and
  // build kernels if this is the first
  if (!ogs::Nrefs) ogs::initKernels(platform);
  ogs::Nrefs++;
}

ogs_t::~ogs_t() {
  Free();
  ogs::Nrefs--;
  if (!ogs::Nrefs) ogs::freeKernels();
}

void ogs_t::Free() {
  //TODO
}

// void setupRowBlocks(ogsData_t &A, platform_t &platform) {

//   dlong blockSum=0;
//   A.NrowBlocks=0;
//   if (A.Nrows) A.NrowBlocks++;
//   for (dlong i=0;i<A.Nrows;i++) {
//     dlong rowSize = A.rowStarts[i+1]-A.rowStarts[i];

//     if (rowSize > ogs::gatherNodesPerBlock) {
//       //this row is pathalogically big. We can't currently run this
//       stringstream ss;
//       ss << "Multiplicity of global node id: " << i << "in ogsSetup is too large.";
//       LIBP_ABORT(ss.str())
//     }

//     if (blockSum+rowSize > ogs::gatherNodesPerBlock) { //adding this row will exceed the nnz per block
//       A.NrowBlocks++; //count the previous block
//       blockSum=rowSize; //start a new row block
//     } else {
//       blockSum+=rowSize; //add this row to the block
//     }
//   }

//   A.blockRowStarts  = (dlong*) calloc(A.NrowBlocks+1,sizeof(dlong));

//   blockSum=0;
//   A.NrowBlocks=0;
//   if (A.Nrows) A.NrowBlocks++;
//   for (dlong i=0;i<A.Nrows;i++) {
//     dlong rowSize = A.rowStarts[i+1]-A.rowStarts[i];

//     if (blockSum+rowSize > ogs::gatherNodesPerBlock) { //adding this row will exceed the nnz per block
//       A.blockRowStarts[A.NrowBlocks++] = i; //mark the previous block
//       blockSum=rowSize; //start a new row block
//     } else {
//       blockSum+=rowSize; //add this row to the block
//     }
//   }
//   A.blockRowStarts[A.NrowBlocks] = A.Nrows;

//   A.o_blockRowStarts = platform.malloc((A.NrowBlocks+1)*sizeof(dlong), A.blockRowStarts);
// }

} //namespace ogs