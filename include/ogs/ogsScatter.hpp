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

#ifndef OGS_SCATTER_HPP
#define OGS_SCATTER_HPP

#include "ogs.hpp"

namespace ogs {

//a Scatter class is transposed Gather. Since all
// Gathers will have at most 1 non-zero per column,
// Scatters have at most 1 non-zero per row. We can
// therefore represent a scatter with just an index
// mapping
class ogsScatter_t {
public:
  dlong Nrows=0;
  dlong Ncols=0;
  dlong *colIds=nullptr;
  occa::memory o_colIds;

  //build a scatter operator from a transposed gather
  ogsScatter_t(ogsGather_t * gather, platform_t &platform);

  void Free();

  void Apply(occa::memory& o_gv, occa::memory& o_v);

  void Apply(dfloat *v, const dfloat *gv);
};

} //namespace ogs

#endif
