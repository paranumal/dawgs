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

void ogsAllToAll_t::Start(occa::memory& o_v){

  const size_t Nbytes = sizeof(dfloat);
  reallocOccaBuffer(Nbytes);

  // assemble mpi send buffer by gathering halo nodes and scattering
  // them into the send buffer
  sendS->Apply(o_v, o_sendBuf);

  dlong Nsend = mpiSendOffsets[size];
  if (Nsend) {
    occa::device &device = platform.device;

    //wait for previous kernel to finish
    device.finish();

    //switch streams to overlap data movement
    occa::stream currentStream = device.getStream();
    device.setStream(dataStream);

    o_sendBuf.copyTo(sendBuf, Nsend*Nbytes, 0, "async: true");

    device.setStream(currentStream);
  }
}


void ogsAllToAll_t::Finish(occa::memory& o_v){

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

  // collect everything needed with single MPI all to all
  MPI_Alltoallv(sendBuf, mpiSendCounts, mpiSendOffsets, MPI_DFLOAT,
                recvBuf, mpiRecvCounts, mpiRecvOffsets, MPI_DFLOAT,
                comm);

  //if we recvieved anything via MPI, gather the recv buffer and scatter
  // it back to to original vector
  dlong Nrecv = mpiRecvOffsets[size];
  if (Nrecv) {
    occa::stream currentStream = device.getStream();
    device.setStream(dataStream);

    // copy recv back to device
    o_recvBuf.copyFrom(recvBuf, Nrecv*Nbytes, 0, "async: true");

    device.finish();
    device.setStream(currentStream);

    recvS->Apply(o_recvBuf, o_v);
  }
}



} //namespace ogs