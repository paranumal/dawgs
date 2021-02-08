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

/*
  OCCA Gather/Scatter Library

  The code

    MPI_Comm comm;
  	dlong N;
    hlong id[N];    // the hlong and dlong types are defined in "types.h"
    int verbose;
    occa::device device
    ...
    ogs_t *ogs = ogs_t::Setup(N, id, comm, verbose, device);

  defines a partition of the set of (processor, local index) pairs,
    (p,i) \in S_j  iff   abs(id[i]) == j  on processor p
  That is, all (p,i) pairs are grouped together (in group S_j) that have the
    same id (=j).
  S_0 is treated specially --- it is ignored completely
    (i.e., when id[i] == 0, local index i does not participate in any
    gather/scatter operation
  If id[i] on proc p is negative then the pair (p,i) is "flagged". This
  determines the non-symmetric behavior. For the simpler, symmetric case,
  all id's should be positive.

  When "ogs" is no longer needed, free it with

    ogsFree(ogs);

  A basic gatherScatter operation is, e.g.,

    occa::memory o_v;
    ...
    ogs->GatherScatter(o_v, ogs_double, ogs_add, ogs_sym);

  This gs call has the effect,

    o_v[i] <--  \sum_{ (p,j) \in S_{id[i]} } o_v_(p) [j]

  where o_v_(p) [j] means o_v[j] on proc p. In other words, every o_v[i] is replaced
  by the sum of all o_v[j]'s with the same id, given by id[i]. This accomplishes
  "direct stiffness summation" corresponding to the action of QQ^T, where
  "Q" is a boolean matrix that copies from a global vector (indexed by id)
  to the local vectors indexed by (p,i) pairs.

  Summation on doubles is not the only operation and datatype supported. Support
  includes the operations
    ogs_add, ogs_mul, ogs_max, ogs_min
  and datatypes
    ogs_dfloat, ogs_double, ogs_float, ogs_int, ogs_longlong, ogs_dlong, ogs_hlong.

  For the nonsymmetric behavior, the "transpose" parameter is important:

    ogs->GatherScatter(o_v, ogs_double, ogs_add, [ogs_notrans/ogs_trans/ogs_sym]);

  When transpose == ogs_notrans, any "flagged" (p,i) pairs (id[i] negative on p)
  do not participate in the sum, but *do* still receive the sum on output.
  As a special case, when only one (p,i) pair is unflagged per group this
  corresponds to the rectangular "Q" matrix referred to above.

  When transpose == ogs_trans, the "flagged" (p,i) pairs *do* participate in the sum,
  but do *not* get set on output. In the special case of only one unflagged
  (p,i) pair, this corresponds to the transpose of "Q" referred to above.

  When transpose == ogs_sym, all ids are considered "unflagged". That is,
  the "flagged" (p,i) pairs *do* participate in the sum, and *do* get set
  on output.

  An additional nonsymmetric operation is

    ogs->Gather(o_Gv, o_v, ogs_double, ogs_add, ogs_notrans);

  this has the effect of "assembling" the vector o_Gv. That is

    o_Gv[gid[j]] <--  \sum_{ (p,j) \in S_{id[i]} } o_v_(p) [j]

  for some ordering gid. As with the GatherScatter operation, when
  transpose == ogs_notrans, any "flagged" (p,i) pairs (id[i] negative on p)
  do not participate in the sum, whereas when transpose == ogs_trans the "flagged"
  (p,i) pairs *do* participate in the sum. Using transpose == ogs_sym is not
  supported (the symmetrized version of this operation is just GatherScatter).

  The reverse of this operation is

    ogs->Scatter(o_v, o_Gv, ogs_double, ogs_add, ogs_notrans);

  which has the effect of scattering in the assembled entries in o_Gv back to the
  orginal ordering. When transpose == ogs_notrans, "flagged" (p,i) pairs (id[i]
  negative on p) recieve their corresponding entry from o_Gv, and when
  transpose == ogs_trans the "flagged" (p,i) pairs do *not* recieve an entry.
  Using transpose == ogs_sym is not supported.

  A versions for vectors (contiguously packed) is, e.g.,

    occa::memory o_v;
    ogs->GatherScatterVec(o_v, k, ogs_double, ogs_add, ogs_sym);

  which is like "GatherScatter" operating on the datatype double[k],
  with summation here being vector summation. Number of messages sent
  is independent of k.

  For combining the communication for "GatherScatter" on multiple arrays:

    occa::memory o_v1, o_v2, ..., o_vk;

    ogs->GatherScatterMany(o_v, k, stride, ogs_double, op, trans);

  when the arrays o_v1, o_v2, ..., o_vk are packed in o_v as

    o_v1 = o_v + 0*stride, o_v2 = o_v + 1*stride, ...

  This call is equivalent to

    ogs->GatherScatter(o_v1, ogs_double, op, trans);
    ogs->GatherScatter(o_v2, ogs_double, op, trans);
    ...
    ogs->GatherScatter(o_vk, ogs_double, op, trans);

  except that all communication is done together.

  A utility function, ogs_t::Unique is provided

    ogs_t::Unique(ids, N, comm);

  This call modifies ids, "flagging" (by negating id[i]) all (p,i) pairs in
  each group except one. The sole "unflagged" member of the group is chosen
  in an arbitrary but consistent way.

  Asynchronous versions of the various GatherScatter functions are provided by

    ogs->GatherScatterStart(o_v, ogs_double, ogs_add, ogs_sym);
    ...
    ogs->GatherScatterFinish(o_v, ogs_double, ogs_add, ogs_sym);

  MPI communication is not initiated in GatherScatterStart, rather some initial
  message packing and host<->device transfers are queued. The user can then queue
  their own local kernels to the device which overlapps with this work before
  calling GatherScatterFinish. The MPI communication will then take place while the
  user's local kernels execute to maximize the amount of communication hiding.

  Finally, a thin wrapper of the ogs_t object, named halo_t is provided. This object
  is intended to provided support for thin halo exchages between MPI procceses.

*/

#ifndef OGS_HPP
#define OGS_HPP

#include "core.hpp"
#include "platform.hpp"

namespace ogs {

/* type enum */
typedef enum { ogs_float, ogs_double, ogs_int, ogs_long_long, ogs_type_n} ogs_type;

/* operation enum */
typedef enum { ogs_add, ogs_mul, ogs_max, ogs_min, ogs_op_n} ogs_op;

/* transpose switch */
typedef enum { ogs_sym, ogs_notrans, ogs_trans } ogs_transpose;

/* method switch */
typedef enum { ogs_auto, ogs_pairwise, ogs_crystal_router, ogs_all_reduce} ogs_method;

//forward declarations
class ogsGather_t;
class ogsScatter_t;
class ogsGatherScatter_t;
class ogsExchange_t;

struct parallelNode_t;

// OCCA+gslib gather scatter
class ogs_t {
public:
  platform_t& platform;
  MPI_Comm comm;

  dlong         N=0;
  dlong         Nlocal=0;         //  number of local nodes
  dlong         Nhalo=0;          //  number of halo nodes

  dlong         Ngather=0;        //  total number of gather nodes
  hlong         NgatherGlobal=0;  //  global number of gather nodes

  ogs_t(platform_t& _platform);
  ~ogs_t();

  void Setup(dlong N, hlong *ids, MPI_Comm comm, int verbose);
  void Free();

  static void Unique(hlong *ids, dlong _N, MPI_Comm _comm);

  // Synchronous device buffer versions
  void GatherScatter    (occa::memory&  o_v, const ogs_method method,
                         const bool gpu_aware, const bool overlap){
    GatherScatterStart (o_v, method, gpu_aware, overlap);
    GatherScatterFinish(o_v, method, gpu_aware, overlap);
  }

  // Asynchronous device buffer versions
  void GatherScatterStart     (occa::memory&  o_v, const ogs_method method,
                               const bool gpu_aware, const bool overlap);
  void GatherScatterFinish    (occa::memory&  o_v, const ogs_method method,
                               const bool gpu_aware, const bool overlap);

private:
  ogsGather_t *gatherLocal=nullptr;
  ogsScatter_t *scatterLocal=nullptr;

  ogsGatherScatter_t *gsLocalS=nullptr;

  ogsGather_t *gatherHalo=nullptr;
  ogsScatter_t *scatterHalo=nullptr;

  ogsExchange_t *exchange_ar=nullptr;
  ogsExchange_t *exchange_pw=nullptr;
  ogsExchange_t *exchange_cr=nullptr;

  void LocalSetup(const dlong Nids, parallelNode_t* nodes,
                  const dlong NbaseIds, dlong *indexMap);
};

} //namespace ogs

#endif
