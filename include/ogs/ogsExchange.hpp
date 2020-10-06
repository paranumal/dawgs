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

namespace ogs {

//virtual base class to perform MPI exchange of gatherScatter
class ogsExchange_t {
public:
  virtual void Start(occa::memory &o_v)=0;
  virtual void Finish(occa::memory &o_v)=0;

  virtual void reallocHostBuffer(size_t Nbytes)=0;
  virtual void reallocOccaBuffer(size_t Nbytes)=0;
};

//MPI communcation via single MPI_Alltoallv call
class ogsAllToAll_t: public ogsExchange_t {
public:
  platform_t &platform;
  MPI_Comm comm;
  int rank, size;

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
  ogsAllToAll_t(platform_t &_platform):
    platform(_platform) {}

  virtual void Start(occa::memory &o_v);
  virtual void Finish(occa::memory &o_v);

  virtual void reallocHostBuffer(size_t Nbytes);
  virtual void reallocOccaBuffer(size_t Nbytes);
};

} //namespace ogs

#endif
