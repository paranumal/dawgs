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

#ifndef OGS_EXCHANGE_HPP
#define OGS_EXCHANGE_HPP

#include "ogs.hpp"
#include "ogs/ogsGather.hpp"
#include "ogs/ogsScatter.hpp"
#include "ogs/ogsGatherScatter.hpp"

namespace ogs {

//virtual base class to perform MPI exchange of gatherScatter
class ogsExchange_t {
public:
  virtual void Start(occa::memory &o_v, bool ga, bool overlap)=0;
  virtual void Finish(occa::memory &o_v, bool ga, bool overlap)=0;

  virtual void reallocOccaBuffer(size_t Nbytes)=0;
};

//MPI communcation via single MPI_Alltoallv call
class ogsAllToAll_t: public ogsExchange_t {
private:
  platform_t &platform;
  MPI_Comm comm;
  int rank, size;

  ogsGather_t *prempi=nullptr, *postmpi=nullptr;

  ogsGatherScatter_t* sendS=nullptr;
  ogsGatherScatter_t* recvS=nullptr;

  void *sendBuf=nullptr, *recvBuf=nullptr;
  occa::memory o_sendBuf, o_recvBuf;
  occa::memory h_sendBuf, h_recvBuf;

  int *mpiSendCounts =nullptr;
  int *mpiRecvCounts =nullptr;
  int *mpiSendOffsets=nullptr;
  int *mpiRecvOffsets=nullptr;

public:
  ogsAllToAll_t(dlong recvN,
               parallelNode_t* recvNodes,
               dlong NgatherLocal,
               ogsGather_t *gatherHalo,
               dlong *indexMap,
               MPI_Comm _comm,
               platform_t &_platform);

  virtual void Start(occa::memory &o_v, bool ga, bool overlap);
  virtual void Finish(occa::memory &o_v, bool ga, bool overlap);

  virtual void reallocOccaBuffer(size_t Nbytes);

};

//MPI communcation via pairwise send/recvs
class ogsPairwise_t: public ogsExchange_t {
private:
  platform_t &platform;
  MPI_Comm comm;
  int rank, size;

  ogsGather_t *prempi=nullptr, *postmpi=nullptr;

  ogsGatherScatter_t* sendS=nullptr;
  ogsGatherScatter_t* recvS=nullptr;

  void *sendBuf=nullptr, *recvBuf=nullptr;
  occa::memory o_sendBuf, o_recvBuf;
  occa::memory h_sendBuf, h_recvBuf;

  int NranksSend=0, NranksRecv=0;
  int *sendRanks =nullptr;
  int *recvRanks =nullptr;
  int *sendCounts =nullptr;
  int *recvCounts =nullptr;
  int *sendOffsets=nullptr;
  int *recvOffsets=nullptr;
  MPI_Request* requests;
  MPI_Status* statuses;

public:
  ogsPairwise_t(dlong recvN,
               parallelNode_t* recvNodes,
               dlong NgatherLocal,
               ogsGather_t *gatherHalo,
               dlong *indexMap,
               MPI_Comm _comm,
               platform_t &_platform);

  virtual void Start(occa::memory &o_v, bool ga, bool overlap);
  virtual void Finish(occa::memory &o_v, bool ga, bool overlap);

  virtual void reallocOccaBuffer(size_t Nbytes);
};

//MPI communcation via Crystal Router
class ogsCrystalRouter_t: public ogsExchange_t {
private:
  platform_t &platform;
  MPI_Comm comm;
  int rank, size;

  ogsGather_t  *gatherHalo=nullptr;
  ogsScatter_t *scatterHalo=nullptr;

  MPI_Request request[3];
  MPI_Status status[3];

  struct crLevel {
    int Nmsg;
    int partner;

    int Nsend, Nrecv0, Nrecv1;

    dlong *sendIds;
    dlong *recvIds0, *recvIds1;
  };

  int Nlevels=0;
  crLevel* levels=nullptr;

  int Nhalo=0, NhaloExt=0;
  void *haloBuf=nullptr;
  occa::memory o_haloBuf, h_haloBuf;

  int NsendMax=0, NrecvMax=0;
  void *sendBuf=nullptr, *recvBuf=nullptr;
  occa::memory o_sendBuf, o_recvBuf;
  occa::memory h_sendBuf, h_recvBuf;

public:
  ogsCrystalRouter_t(dlong recvN,
               parallelNode_t* recvNodes,
               dlong NgatherLocal,
               ogsGather_t *_gatherHalo,
               dlong *indexMap,
               MPI_Comm _comm,
               platform_t &_platform);

  virtual void Start(occa::memory &o_v, bool ga, bool overlap);
  virtual void Finish(occa::memory &o_v, bool ga, bool overlap);

  virtual void reallocOccaBuffer(size_t Nbytes);
};

//MPI communcation via binary tree
class ogsBinaryTree_t: public ogsExchange_t {
private:
  platform_t &platform;
  MPI_Comm comm;
  int rank, size;

  ogsGather_t  *gatherHalo=nullptr;
  ogsScatter_t *scatterHalo=nullptr;

  ogsGather_t  *partialGather=nullptr;
  ogsScatter_t *partialScatter=nullptr;
  ogsGatherScatter_t* rootGS=nullptr;

  void *sBuf=nullptr, *gBuf=nullptr;
  occa::memory o_sBuf, o_gBuf;
  occa::memory h_sBuf, h_gBuf;

  int Npartners=0;
  int upstreamPartner;
  int *downstreamPartners =nullptr;

  int sTotal=0, gTotal=0, Nsend=0;
  int *sCounts =nullptr;
  int *sOffsets=nullptr;
  MPI_Request* requests;
  MPI_Status* statuses;

public:
  ogsBinaryTree_t(dlong recvN,
               parallelNode_t* recvNodes,
               dlong NgatherLocal,
               ogsGather_t *_gatherHalo,
               dlong *indexMap,
               MPI_Comm _comm,
               platform_t &_platform);

  virtual void Start(occa::memory &o_v, bool ga, bool overlap);
  virtual void Finish(occa::memory &o_v, bool ga, bool overlap);

  virtual void reallocOccaBuffer(size_t Nbytes);
};

} //namespace ogs

#endif
