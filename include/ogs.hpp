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
    ogs_t ogs(platform);
    ...
    ogs.Setup(N, id, comm, verbose, device);

  defines a partition of the set of (processor, local index) pairs,
    (p,i) \in S_j  iff   abs(id[i]) == j  on processor p
  That is, all (p,i) pairs are grouped together (in group S_j) that have the
    same id (=j).
  S_0 is treated specially --- it is ignored completely
    (i.e., when id[i] == 0, local index i does not participate in any
    gather/scatter operation)
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
typedef enum { Float, Double, Int32, Int64} Type;

constexpr Type Dfloat = (std::is_same<double, dfloat>::value)
                          ? Double : Float;
constexpr Type Dlong  = (std::is_same<int32_t, dlong>::value)
                          ? Int32 : Int64;
constexpr Type Hlong  = (std::is_same<int32_t, hlong>::value)
                          ? Int32 : Int64;

/* operation enum */
typedef enum { Add, Mul, Max, Min} Op;

/* transpose switch */
typedef enum { Sym, NoTrans, Trans } Transpose;

/* method switch */
typedef enum { Auto, Pairwise, CrystalRouter, AllToAll} Method;

/* kind enum */
typedef enum { Unsigned, Signed, Halo} Kind;

}

#include "ogs/ogsBase.hpp"

namespace ogs {

//pre-build kernels
void InitializeKernels(platform_t& platform, const Type type, const Op op);

// OCCA Gather Scatter
class ogs_t : public ogsBase_t {
public:
  ogs_t(platform_t& _platform):
   ogsBase_t(_platform) {}

  void Setup(const dlong _N,
             hlong *ids,
             MPI_Comm _comm,
             const Kind _kind,
             const Method method,
             const bool _unique,
             const bool verbose);
  void Free();

  void SetupGlobalToLocalMapping(dlong *GlobalToLocal);

  // host versions
  void GatherScatter(void* v,
                     const int k,
                     const Type type,
                     const Op op,
                     const Transpose trans);
  // Synchronous device buffer versions
  void GatherScatter(occa::memory&  o_v,
                     const int k,
                     const Type type,
                     const Op op,
                     const Transpose trans);
  // Asynchronous device buffer versions
  void GatherScatterStart (occa::memory&  o_v,
                           const int k,
                           const Type type,
                           const Op op,
                           const Transpose trans);
  void GatherScatterFinish(occa::memory&  o_v,
                           const int k,
                           const Type type,
                           const Op op,
                           const Transpose trans);

  // host versions
  void Gather(void* gv,
              const void* v,
              const int k,
              const Type type,
              const Op op,
              const Transpose trans);
  // Synchronous device buffer versions
  void Gather(occa::memory&  o_gv,
              occa::memory&  o_v,
              const int k,
              const Type type,
              const Op op,
              const Transpose trans);
  // Asynchronous device buffer versions
  void GatherStart (occa::memory&  o_gv,
                    occa::memory&  o_v,
                    const int k,
                    const Type type,
                    const Op op,
                    const Transpose trans);
  void GatherFinish(occa::memory&  o_gv,
                    occa::memory&  o_v,
                    const int k,
                    const Type type,
                    const Op op,
                    const Transpose trans);

  // host versions
  void Scatter(void* v,
               const void* gv,
               const int k,
               const Type type,
               const Op op,
               const Transpose trans);
  // Synchronous device buffer versions
  void Scatter(occa::memory&  o_v,
               occa::memory&  o_gv,
               const int k,
               const Type type,
               const Op op,
               const Transpose trans);
  // Asynchronous device buffer versions
  void ScatterStart (occa::memory&  o_v,
                     occa::memory&  o_gv,
                     const int k,
                     const Type type,
                     const Op op,
                     const Transpose trans);
  void ScatterFinish(occa::memory&  o_v,
                     occa::memory&  o_gv,
                     const int k,
                     const Type type,
                     const Op op,
                     const Transpose trans);

  friend class halo_t;
};

// OCCA Halo
class halo_t : public ogsBase_t {
public:
  halo_t(platform_t& _platform):
   ogsBase_t(_platform) {}

  bool gathered_halo=false;
  dlong Nhalo=0;

  void Setup(const dlong _N,
             hlong *ids,
             MPI_Comm _comm,
             const Method method,
             const bool verbose);

  void SetupFromGather(ogs_t& ogs);

  void Free();

  // Host version
  void Exchange(void  *v, const int k, const Type type);
  // Synchronous device buffer version
  void Exchange(occa::memory &o_v, const int k, const Type type);
  // Asynchronous device buffer version
  void ExchangeStart (occa::memory &o_v, const int k, const Type type);
  void ExchangeFinish(occa::memory &o_v, const int k, const Type type);

  // Host version
  void Combine(void  *v, const int k, const Type type);
  // Synchronous device buffer version
  void Combine(occa::memory &o_v, const int k, const Type type);
  // Asynchronous device buffer version
  void CombineStart (occa::memory &o_v, const int k, const Type type);
  void CombineFinish(occa::memory &o_v, const int k, const Type type);
};

} //namespace ogs

#endif
