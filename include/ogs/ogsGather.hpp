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

#ifndef OGS_GATHER_HPP
#define OGS_GATHER_HPP

#include "ogs.hpp"

namespace ogs {

//a Gather class is essentially a sparse CSR matrix,
// with no vals stored. By construction, all Gather
// structs will have at most 1 non-zero per column.
// We specialize the case where the Gather is diagonal,
// i.e. has 1 non-zero per row.
class ogsGather_t {
public:
  dlong nnz=0;
  dlong Nrows=0;
  dlong Ncols=0;
  dlong *rowStarts=nullptr;
  dlong *colIds=nullptr;

  occa::memory o_rowStarts;
  occa::memory o_colIds;

  dlong NrowBlocks=0;
  dlong *blockRowStarts=nullptr;
  occa::memory o_blockRowStarts;

  bool is_diag=false;

  void Free();
  ~ogsGather_t() {Free();}

  void Apply(occa::memory& o_gv, occa::memory& o_v);

  void Apply(dfloat *gv, const dfloat* v);

  void setupRowBlocks(platform_t &platform);
};


} //namespace ogs

#endif
