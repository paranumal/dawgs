
Device AWare Gather Scatter (DAWGS)
===================================

This is the `dawgs` repository.  `dawgs` is the testbed development of a
GPU-enabled gather scatter library and experimentation of GPU-aware MPI.

The problem setup is a uniform NX x NY x NZ mesh of hexahehral elements,
one which a global node ordering is constructed and a gather-scatter
operation is run and benchmarked for several MPI communication algorithms.

How to compile `dawgs`
------------------------

There are a couple of prerequisites for building `dawgs`;

- An MPI stack.  Any will work;
- ROCm version 3.5 or newer;
- OpenBlas.

Installing `MPI` and `OpenBlas` can be done using whatever package manager your
operating system provides.

You will need to install `ROCm`, which can be done using the system package
manager once you have added the appropriate repository to the package manager.

To build and run `dawgs`:

    $ git clone --recursive <dawgs repo>
    $ cd /path/to/dawgs
    $ export OPENBLAS_DIR=/path/to/openblas
    $ make -j `nproc`

How to run `dawgs`
--------------------

Here is an example CORAL-2 problem size that you can run on one GPU:

    $ mpirun -np 1 ./dawgsMain -m HIP -nx 24 -ny 24 -nz 24 -p 3

Here is the meaning of each of the command line options

- `nx`: the number of spectral elements in the x-direction per MPI rank
- `ny`: the number of spectral elements in the y-direction per MPI rank
- `nz`: the number of spectral elements in the z-direction per MPI rank
- `p`: the order of the polynomial (the number of nodes per hex is (p+1)^3)
- `m`: the mode to run OCCA in, `HIP` is for AMD GPUs but `CUDA` and `Serial`
are also supported

Running on multiple GPUs can by done by passing a larger argument to `np` and
specifying the number of MPI ranks in each coordinate direction:

    $ mpirun -np 2 ./dawgsMain -m HIP -nx 24 -ny 24 -nz 24 -p 3

Verifying correctness
---------------------

To verify that the computation is correct, an optional `-cc` argument can be
passed. Upen doing so, the result from a gather-scatter is compared with a
legacy implementation using gslib:

    $ mpirun -np 2 ./dawgsMain -m CUDA -nx 10 -ny 10 -nz 10 -p 3 -cc
    Ranks = 2, Global DOFS = 128000, Max Local DOFS = 64000, Degree = 3
    AllToAll Method , Error = 0
    Pairwise Method , Error = 0
    Crystal Router Method , Error = 0
    AllToAll Method , Halo Kernel Overlap , Error = 0
    Pairwise Method , Halo Kernel Overlap , Error = 0
    Crystal Router Method , Halo Kernel Overlap , Error = 0

All `Errors` should report zero.

Scaling Studies
---------------

An optional `-sw` argument can be passed to run a collection of problem sizes
and polynomial degrees for quickly collecting scaling data. Other input
parameters are typically not required for this mode:

    $ ./dawgsMain -m CUDA -cc
    AR : Ranks = 2, Global DOFS = 128, Max Local DOFS = 64, Degree = 1, Nvectors = 0, Time taken = 0.350167 ms, DOFS/s = 365540, DOFS/(s*rank) = 182770
    PW : Ranks = 2, Global DOFS = 128, Max Local DOFS = 64, Degree = 1, Nvectors = 0, Time taken = 0.432008 ms, DOFS/s = 296291, DOFS/(s*rank) = 148145
    CR : Ranks = 2, Global DOFS = 128, Max Local DOFS = 64, Degree = 1, Nvectors = 0, Time taken = 0.329541 ms, DOFS/s = 388419, DOFS/(s*rank) = 194210
    AR , Overlap : Ranks = 2, Global DOFS = 128, Max Local DOFS = 64, Degree = 1, Nvectors = 0, Time taken = 0.385979 ms, DOFS/s = 331624, DOFS/(s*rank) = 165812
    PW , Overlap : Ranks = 2, Global DOFS = 128, Max Local DOFS = 64, Degree = 1, Nvectors = 0, Time taken = 0.381121 ms, DOFS/s = 335851, DOFS/(s*rank) = 167925
    CR , Overlap : Ranks = 2, Global DOFS = 128, Max Local DOFS = 64, Degree = 1, Nvectors = 0, Time taken = 0.32478 ms, DOFS/s = 394113, DOFS/(s*rank) = 197056
    AR : Ranks = 2, Global DOFS = 16000, Max Local DOFS = 8000, Degree = 1, Nvectors = 0, Time taken = 0.389301 ms, DOFS/s = 4.10993e+07, DOFS/(s*rank) = 2.05496e+07
    PW : Ranks = 2, Global DOFS = 16000, Max Local DOFS = 8000, Degree = 1, Nvectors = 0, Time taken = 0.359486 ms, DOFS/s = 4.4508e+07, DOFS/(s*rank) = 2.2254e+07
    CR : Ranks = 2, Global DOFS = 16000, Max Local DOFS = 8000, Degree = 1, Nvectors = 0, Time taken = 0.303428 ms, DOFS/s = 5.27309e+07, DOFS/(s*rank) = 2.63654e+07
    AR , Overlap : Ranks = 2, Global DOFS = 16000, Max Local DOFS = 8000, Degree = 1, Nvectors = 0, Time taken = 0.36022 ms, DOFS/s = 4.44173e+07, DOFS/(s*rank) = 2.22087e+07
    PW , Overlap : Ranks = 2, Global DOFS = 16000, Max Local DOFS = 8000, Degree = 1, Nvectors = 0, Time taken = 0.358692 ms, DOFS/s = 4.46066e+07, DOFS/(s*rank) = 2.23033e+07
    CR , Overlap : Ranks = 2, Global DOFS = 16000, Max Local DOFS = 8000, Degree = 1, Nvectors = 0, Time taken = 0.307693 ms, DOFS/s = 5.2e+07, DOFS/(s*rank) = 2.6e+07
    AR : Ranks = 2, Global DOFS = 93312, Max Local DOFS = 46656, Degree = 1, Nvectors = 0, Time taken = 0.330405 ms, DOFS/s = 2.82417e+08, DOFS/(s*rank) = 1.41208e+08
    PW : Ranks = 2, Global DOFS = 93312, Max Local DOFS = 46656, Degree = 1, Nvectors = 0, Time taken = 0.375525 ms, DOFS/s = 2.48484e+08, DOFS/(s*rank) = 1.24242e+08
    CR : Ranks = 2, Global DOFS = 93312, Max Local DOFS = 46656, Degree = 1, Nvectors = 0, Time taken = 0.356722 ms, DOFS/s = 2.61582e+08, DOFS/(s*rank) = 1.30791e+08
    ...

How to clean build objects
--------------------------

To clean the `dawgs` build objects:

    $ cd /path/to/dawgs/repo
    $ make realclean

To clean JIT kernel objects:

    $ cd /path/to/dawgs/repo
    $ rm -r .occa

Please invoke `make help` for more supported options.
