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

namespace ogs {

void ogsGather_t::Apply(occa::memory&  o_gv, occa::memory&  o_v) {
  if (NrowBlocks)
    gatherKernel_double_add(NrowBlocks,
                             o_blockRowStarts,
                             o_rowStarts,
                             o_colIds,
                             o_v,
                             o_gv);
}

void ogsGather_t::Apply(dfloat *gv, const dfloat *v) {

  for(dlong n=0;n<Nrows;++n){
    const dlong start = rowStarts[n];
    const dlong end   = rowStarts[n+1];

    dfloat val = 0.0;
    for(dlong g=start;g<end;++g){
      val += v[colIds[g]];
    }
    gv[n] = val;
  }
}

void ogsGather_t::setupRowBlocks(platform_t &platform) {

  dlong blockSum=0;
  NrowBlocks=0;
  if (Nrows) NrowBlocks++;
  for (dlong i=0;i<Nrows;i++) {
    const dlong rowSize  = rowStarts[i+1]-rowStarts[i];

    if (rowSize > ogs::gatherNodesPerBlock) {
      //this row is pathalogically big. We can't currently run this
      stringstream ss;
      ss << "Multiplicity of global node id: " << i << "in ogsGather_t::setupRowBlocks is too large.";
      LIBP_ABORT(ss.str())
    }

    if (blockSum+rowSize > ogs::gatherNodesPerBlock) { //adding this row will exceed the nnz per block
      NrowBlocks++; //count the previous block
      blockSum=rowSize; //start a new row block
    } else {
      blockSum+=rowSize; //add this row to the block
    }
  }

  blockRowStarts  = (dlong*) calloc(NrowBlocks+1,sizeof(dlong));

  blockSum=0;
  NrowBlocks=0;
  if (Nrows) NrowBlocks++;
  for (dlong i=0;i<Nrows;i++) {
    const dlong rowSize  = rowStarts[i+1]-rowStarts[i];

    if (blockSum+rowSize > ogs::gatherNodesPerBlock) { //adding this row will exceed the nnz per block
      blockRowStarts[NrowBlocks++] = i; //mark the previous block
      blockSum=rowSize; //start a new row block
    } else {
      blockSum+=rowSize; //add this row to the block
    }
  }
  blockRowStarts[NrowBlocks] = Nrows;

  o_blockRowStarts = platform.malloc((NrowBlocks+1)*sizeof(dlong), blockRowStarts);
}

} //namespace ogs