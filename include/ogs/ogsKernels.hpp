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

#ifndef OGS_KERNELS_HPP
#define OGS_KERNELS_HPP

#include <limits>
#include "ogs.hpp"
#include "ogsDefs.h"

#define DEFINE_ADD_OGS_INIT(T)                                  \
  static T init_##T##_add = (T)  0;                             \
  static T init_##T##_mul = (T)  1;                             \
  static T init_##T##_min = (T)  std::numeric_limits<T>::max(); \
  static T init_##T##_max = (T) -std::numeric_limits<T>::max();

class ogsData_t;

namespace ogs {

extern const int blockSize;
extern const int gatherNodesPerBlock;

extern int Nrefs;

extern occa::stream dataStream;

struct parallelNode_t{

  dlong localId;    // local node id
  hlong baseId;     // original global index

  dlong newId;         // new global id
  int sign;

  int rank; //original rank
  int destRank; //destination rank

};

extern MPI_Datatype MPI_PARALLELNODE_T;

void initKernels(platform_t& platform);
void freeKernels();

#define DEFINE_GATHERSCATTER_KERNEL(T,OP) \
  extern occa::kernel gatherScatterKernel_##T##_##OP;

#define DEFINE_GATHER_KERNEL(T,OP) \
  extern occa::kernel gatherKernel_##T##_##OP;

#define DEFINE_SCATTER_KERNEL(T)              \
  extern occa::kernel scatterKernel_flat_##T; \
  extern occa::kernel scatterKernel_##T;

#define DEFINE_KERNELS(T)                        \
  OGS_FOR_EACH_OP(T,DEFINE_GATHERSCATTER_KERNEL) \
  OGS_FOR_EACH_OP(T,DEFINE_GATHER_KERNEL)        \
  DEFINE_SCATTER_KERNEL(T)

OGS_FOR_EACH_TYPE(DEFINE_KERNELS)

#undef DEFINE_GATHERSCATTER_KERNEL
#undef DEFINE_GATHER_KERNEL
#undef DEFINE_SCATTER_KERNEL
#undef DEFINE_KERNELS

  extern occa::kernel setKernel;
  extern occa::kernel extractKernel;
  extern occa::kernel injectKernel;

} //namespace ogs

#endif
