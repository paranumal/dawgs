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
#include "ogs/ogsGather.hpp"
#include "ogs/ogsScatter.hpp"
#include "ogs/ogsGatherScatter.hpp"
#include "ogs/ogsExchange.hpp"

namespace ogs {

void ogs_t::GatherScatterStart(occa::memory& o_v,
                               const ogs_method method,
                               const bool gpu_aware,
                               const bool overlap){
  //prepare MPI exchange
  if (method == ogs_all_reduce)
    exchange_ar->Start(o_v, gpu_aware, overlap);
  else if (method == ogs_pairwise)
    exchange_pw->Start(o_v, gpu_aware, overlap);
  else
    exchange_cr->Start(o_v, gpu_aware, overlap);
}


void ogs_t::GatherScatterFinish(occa::memory& o_v,
                                const ogs_method method,
                                const bool gpu_aware,
                                const bool overlap){

  //queue local gs operation
  gsLocalS->Apply(o_v);

  //finish MPI exchange
  if (method == ogs_all_reduce)
    exchange_ar->Finish(o_v, gpu_aware, overlap);
  else if (method == ogs_pairwise)
    exchange_pw->Finish(o_v, gpu_aware, overlap);
  else
    exchange_cr->Finish(o_v, gpu_aware, overlap);
}

} //namespace ogs