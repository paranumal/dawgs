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
#include "ogs/ogsGather.hpp"
#include "ogs/ogsScatter.hpp"

namespace ogs {

void ogsScatter_t::Apply(occa::memory& o_v, occa::memory& o_gv) {
  if (Nrows)
    scatterKernel_double(Nrows,
                         o_colIds,
                         o_gv,
                         o_v);
}

void ogsScatter_t::Apply(dfloat *v, const dfloat *gv) {

  for(dlong n=0;n<Nrows;++n){
    const dlong id = colIds[n];
    if (id>=0) v[n] = gv[id];
  }
}

ogsScatter_t::ogsScatter_t(ogsGather_t *gather, platform_t &platform) {
  Nrows = gather->Ncols;
  Ncols = gather->Nrows;

  if (gather->Nrows) {
    colIds = (dlong*) malloc(Nrows*sizeof(dlong));

    for (dlong n=0;n<Nrows;n++)
      colIds[n] = -1;

    for (dlong i=0;i<gather->Nrows;i++) {
      const dlong start = gather->rowStarts[i];
      const dlong end   = gather->rowStarts[i+1];
      for (dlong j=start;j<end;j++) {
        const dlong colId = gather->colIds[j];
        colIds[colId] = i;
      }
    }

    o_colIds = platform.malloc(Nrows*sizeof(dlong), colIds);
  }
}

} //namespace ogs