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

#ifndef OGS_GATHERSCATTER_HPP
#define OGS_GATHERSCATTER_HPP

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
  dlong *rowStarts=nullptr;
  dlong *colIds=nullptr;

  occa::memory o_rowStarts;
  occa::memory o_colIds;

  dlong NrowBlocks=0;
  dlong *blockRowStarts=nullptr;
  occa::memory o_blockRowStarts;

  bool is_diag=false;

  void Free();

  void Apply(occa::memory& o_gv, occa::memory& o_v);
};

//a Scatter class is transposed Gather. Since all
// Gathers will have at most 1 non-zero per column,
// Scatters have at most 1 non-zero per row. We can
// therefore represent a scatter with just an index
// mapping
class ogsScatter_t {
public:
  dlong Nrows=0;
  dlong *colIds=nullptr;
  occa::memory o_colIds;

  void Free();

  void Apply(occa::memory& o_v, occa::memory& o_gv);
};

//a GatherScatter class is the composition of a Gather
// followed by a Scatter (transposed Gather).
class ogsGatherScatter_t {
public:
  dlong Nrows=0;
  ogsGather_t* gather;
  ogsGather_t* scatter;

  dlong NrowBlocks=0;
  dlong *blockRowStarts=nullptr;
  occa::memory o_blockRowStarts;

  bool is_diag=false;

  void Free();

  void Apply(occa::memory& o_v);
  void Apply(occa::memory& o_v, occa::memory& o_w);
};

} //namespace ogs

#endif
