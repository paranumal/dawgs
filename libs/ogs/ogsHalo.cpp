/*

The MIT License (MIT)

Copyright (c) 2017-2021 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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
#include "ogs/ogsUtils.hpp"
#include "ogs/ogsOperator.hpp"
#include "ogs/ogsExchange.hpp"

namespace libp {

namespace ogs {

/********************************
 * Device Exchange
 ********************************/
void halo_t::Exchange(occa::memory& o_v,
                      const int k,
                      const Type type) {
  ExchangeStart (o_v, k, type);
  ExchangeFinish(o_v, k, type);
}

void halo_t::ExchangeStart(occa::memory& o_v,
                           const int k,
                           const Type type){
  exchange->AllocBuffer(k*Sizeof(type));

  occa::memory o_haloBuf = exchange->o_workspace;

  if (exchange->gpu_aware) {
    if (gathered_halo) {
      //if this halo was build from a gathered ogs the halo nodes are at the end
      o_haloBuf.copyFrom(o_v + k*NlocalT*Sizeof(type),
                         k*NhaloP*Sizeof(type),
                         0, 0, "async: true");
    } else {
      //collect halo buffer
      gatherHalo->Gather(o_haloBuf, o_v, k, type, Add, NoTrans);
    }

    //prepare MPI exchange
    exchange->Start(o_haloBuf, k, type, Add, NoTrans);

  } else {
    //get current stream
    occa::device &device = platform.device;
    occa::stream currentStream = device.getStream();

    //if not using gpu-aware mpi move the halo buffer to the host
    memory<dfloat> haloBuf = exchange->h_workspace;

    if (gathered_halo) {
      //wait for o_v to be ready
      device.finish();

      //queue copy to host
      device.setStream(dataStream);
      const size_t Nbytes = k*Sizeof(type);
      o_v.copyTo(haloBuf.ptr(), NhaloP*Nbytes,
                 k*NlocalT*Sizeof(type), "async: true");
      device.setStream(currentStream);
    } else {
      //collect halo buffer
      gatherHalo->Gather(o_haloBuf, o_v, k, type, Add, NoTrans);

      //wait for o_haloBuf to be ready
      device.finish();

      //queue copy to host
      device.setStream(dataStream);
      const size_t Nbytes = k*Sizeof(type);
      o_haloBuf.copyTo(haloBuf.ptr(), NhaloP*Nbytes,
                       0, "async: true");
      device.setStream(currentStream);
    }
  }
}

void halo_t::ExchangeFinish(occa::memory& o_v,
                            const int k,
                            const Type type){

  occa::memory o_haloBuf = exchange->o_workspace;

  //write exchanged halo buffer back to vector
  if (exchange->gpu_aware) {
    //finish MPI exchange
    exchange->Finish(o_haloBuf, k, type, Add, NoTrans);

    if (gathered_halo) {
      o_v.copyFrom(o_haloBuf,
                   k*Nhalo*Sizeof(type),
                   k*(NlocalT+NhaloP)*Sizeof(type),
                   k*NhaloP*Sizeof(type),
                   "async: true");
    } else {
      gatherHalo->Scatter(o_v, o_haloBuf, k, type, NoTrans);
    }
  } else {
    memory<dfloat> haloBuf = exchange->h_workspace;

    //get current stream
    occa::device &device = platform.device;
    occa::stream currentStream = device.getStream();

    //synchronize data stream to ensure the buffer is on the host
    device.setStream(dataStream);
    device.finish();

    /*MPI exchange of host buffer*/
    exchange->Start (haloBuf, k, Add, NoTrans);
    exchange->Finish(haloBuf, k, Add, NoTrans);

    const size_t Nbytes = k*Sizeof(type);
    // copy recv back to device
    if (gathered_halo) {
      o_v.copyFrom(haloBuf.ptr()+NhaloP*Nbytes,
                   Nhalo*Nbytes,
                   k*(NlocalT+NhaloP)*Sizeof(type), "async: true");
      device.finish(); //wait for transfer to finish
      device.setStream(currentStream);
    } else {
      o_haloBuf.copyFrom(haloBuf.ptr(), Nhalo*Nbytes,
                         NhaloP*Nbytes, "async: true");
      device.finish(); //wait for transfer to finish
      device.setStream(currentStream);

      gatherHalo->Scatter(o_v, o_haloBuf, k, type, NoTrans);
    }
  }
}

//host version
template<typename T>
void halo_t::Exchange(memory<T> v, const int k) {
  ExchangeStart (v, k);
  ExchangeFinish(v, k);
}

template<typename T>
void halo_t::ExchangeStart(memory<T> v, const int k) {
  exchange->AllocBuffer(k*sizeof(T));

  memory<T> haloBuf = exchange->h_workspace;

  //collect halo buffer
  if (gathered_halo) {
    //if this halo was build from a gathered ogs the halo nodes are at the end
    haloBuf.copyFrom(v + k*NlocalT, k*NhaloP);
  } else {
    gatherHalo->Gather(haloBuf, v, k, Add, NoTrans);
  }

  //Prepare MPI exchange
  exchange->Start(haloBuf, k, Add, NoTrans);
}

template<typename T>
void halo_t::ExchangeFinish(memory<T> v, const int k) {

  memory<T> haloBuf = exchange->h_workspace;

  //finish MPI exchange
  exchange->Finish(haloBuf, k, Add, NoTrans);

  //write exchanged halo buffer back to vector
  if (gathered_halo) {
    //if this halo was build from a gathered ogs the halo nodes are at the end
    haloBuf.copyTo(v + k*(NlocalT+NhaloP),
                   k*Nhalo,
                   k*NhaloP);
  } else {
    gatherHalo->Scatter(v, haloBuf, k, NoTrans);
  }
}

template void halo_t::Exchange(memory<float> v, const int k);
template void halo_t::Exchange(memory<double> v, const int k);
template void halo_t::Exchange(memory<int> v, const int k);
template void halo_t::Exchange(memory<long long int> v, const int k);

/********************************
 * Combine
 ********************************/
void halo_t::Combine(occa::memory& o_v,
                     const int k,
                     const Type type) {
  CombineStart (o_v, k, type);
  CombineFinish(o_v, k, type);
}

void halo_t::CombineStart(occa::memory& o_v,
                          const int k,
                          const Type type){
  exchange->AllocBuffer(k*Sizeof(type));

  occa::memory o_haloBuf = exchange->o_workspace;

  if (exchange->gpu_aware) {
    if (gathered_halo) {
      //if this halo was build from a gathered ogs the halo nodes are at the end
      o_haloBuf.copyFrom(o_v + k*NlocalT*Sizeof(type),
                         k*NhaloT*Sizeof(type),
                         0, 0, "async: true");
    } else {
      //collect halo buffer
      gatherHalo->Gather(o_haloBuf, o_v, k, type, Add, Trans);
    }

    //prepare MPI exchange
    exchange->Start(o_haloBuf, k, type, Add, Trans);
  } else {
    //get current stream
    occa::device &device = platform.device;
    occa::stream currentStream = device.getStream();

    //if not using gpu-aware mpi move the halo buffer to the host
    memory<dfloat> haloBuf = exchange->h_workspace;

    if (gathered_halo) {
      //wait for o_v to be ready
      device.finish();

      //queue copy to host
      device.setStream(dataStream);
      const size_t Nbytes = k*Sizeof(type);
      o_v.copyTo(haloBuf.ptr(), NhaloT*Nbytes,
                 k*NlocalT*Sizeof(type), "async: true");
      device.setStream(currentStream);
    } else {
      //collect halo buffer
      gatherHalo->Gather(o_haloBuf, o_v, k, type, Add, Trans);

      //wait for o_haloBuf to be ready
      device.finish();

      //queue copy to host
      device.setStream(dataStream);
      const size_t Nbytes = k*Sizeof(type);
      o_haloBuf.copyTo(haloBuf.ptr(), NhaloT*Nbytes,
                       0, "async: true");
      device.setStream(currentStream);
    }
  }
}


void halo_t::CombineFinish(occa::memory& o_v,
                           const int k,
                           const Type type){

  occa::memory o_haloBuf = exchange->o_workspace;

  //write exchanged halo buffer back to vector
  if (exchange->gpu_aware) {
    //finish MPI exchange
    exchange->Finish(o_haloBuf, k, type, Add, Trans);

    if (gathered_halo) {
      //if this halo was build from a gathered ogs the halo nodes are at the end
      o_haloBuf.copyTo(o_v + k*NlocalT*Sizeof(type),
                       k*NhaloP*Sizeof(type),
                       0, 0, "async: true");
    } else {
      gatherHalo->Scatter(o_v, o_haloBuf, k, type, Trans);
    }
  } else {
    memory<dfloat> haloBuf = exchange->h_workspace;

    //get current stream
    occa::device &device = platform.device;
    occa::stream currentStream = device.getStream();

    //synchronize data stream to ensure the buffer is on the host
    device.setStream(dataStream);
    device.finish();

    /*MPI exchange of host buffer*/
    exchange->Start (haloBuf, k, Add, Trans);
    exchange->Finish(haloBuf, k, Add, Trans);

    const size_t Nbytes = k*Sizeof(type);
    if (gathered_halo) {
      // copy recv back to device
      o_v.copyFrom(haloBuf.ptr(), NhaloP*Nbytes,
                   k*NlocalT*Sizeof(type), "async: true");
      device.finish(); //wait for transfer to finish
      device.setStream(currentStream);
    } else {
      o_haloBuf.copyFrom(haloBuf.ptr(), NhaloP*Nbytes,
                         0, "async: true");
      device.finish(); //wait for transfer to finish
      device.setStream(currentStream);

      gatherHalo->Scatter(o_v, o_haloBuf, k, type, Trans);
    }
  }
}

//host version
template<typename T>
void halo_t::Combine(memory<T> v, const int k) {
  CombineStart (v, k);
  CombineFinish(v, k);
}

template<typename T>
void halo_t::CombineStart(memory<T> v, const int k) {
  exchange->AllocBuffer(k*sizeof(T));

  memory<T> haloBuf = exchange->h_workspace;

  //collect halo buffer
  if (gathered_halo) {
    //if this halo was build from a gathered ogs the halo nodes are at the end
    haloBuf.copyFrom(v + k*NlocalT, k*NhaloT);
  } else {
    gatherHalo->Gather(haloBuf, v, k, Add, Trans);
  }

  //Prepare MPI exchange
  exchange->Start(haloBuf, k, Add, Trans);
}


template<typename T>
void halo_t::CombineFinish(memory<T> v, const int k) {

  memory<T> haloBuf = exchange->h_workspace;

  //finish MPI exchange
  exchange->Finish(haloBuf, k, Add, Trans);

  //write exchanged halo buffer back to vector
  if (gathered_halo) {
    //if this halo was build from a gathered ogs the halo nodes are at the end
    haloBuf.copyTo(v + k*NlocalT, k*NhaloP);
  } else {
    gatherHalo->Scatter(v, haloBuf, k, Trans);
  }
}

template void halo_t::Combine(memory<float> v, const int k);
template void halo_t::Combine(memory<double> v, const int k);
template void halo_t::Combine(memory<int> v, const int k);
template void halo_t::Combine(memory<long long int> v, const int k);

} //namespace ogs

} //namespace libp
