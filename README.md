[![DOI](https://zenodo.org/badge/468400696.svg)](https://zenodo.org/badge/latestdoi/468400696)

Device AWare Gather Scatter (DAWGS)
===================================

`dawgs` is the testbed development of a GPU-enabled gather scatter library and experimentation of GPU-aware MPI.

The problem setup is a uniform NX x NY x NZ mesh of hexahehral elements, one which a global node ordering is constructed and a gather-scatter operation is run and benchmarked for several MPI communication algorithms.

How to compile `dawgs`
------------------------

There are a couple of prerequisites for building `dawgs`;

- An MPI stack.  Any will work;

Installing `MPI` can be done using whatever package manager your operating system provides.

To build and run `dawgs`:

    $ git clone --recursive <dawgs repo>
    $ cd /path/to/dawgs
    $ make -j `nproc`

To build and run `dawgs` with GPU-aware MPI support:

    $ make -j `nproc` gpu-aware-mpi=true


How to run `dawgs`
--------------------

Here is an example problem size that you can run on one GPU:

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

To verify that the computation is correct, an optional `-cc TRUE` argument can be
passed. Upen doing so, the result from a gather-scatter is compared with a
legacy implementation using gslib:

    $ mpirun -np 2 ./dawgsMain -m CUDA -nx 10 -ny 10 -nz 10 -p 3 -cc TRUE
    ...
    Device GatherScatter:  Error = 0
    Host   GatherScatter:  Error = 0
    Device GatherScatter:  Error = 0
    Host   GatherScatter:  Error = 0
    Device Gather+Scatter: Error = 0
    Host   Gather+Scatter: Error = 0

All `Errors` should report zero, or near zero.

Scaling Studies
---------------

An optional `-sw` argument can be passed to run a collection of problem sizes
and polynomial degrees for quickly collecting scaling data. Other input
parameters are typically not required for this mode:

    $ ./dawgsMain -m CUDA -sw TRUE
    ...
    Host GatherScatter:   Ranks = 1, Global DOFS = 42875, Max Local DOFS = 314432, Degree = 1, Time taken = 0.0017945 s, DOFS/s = 2.38924e+07, DOFS/(s*rank) = 2.38924e+07
    Device GatherScatter: Ranks = 1, Global DOFS = 42875, Max Local DOFS = 314432, Degree = 1, Time taken = 1.94e-05 s, DOFS/s = 2.21005e+09, DOFS/(s*rank) = 2.21005e+09
    Host Gather:          Ranks = 1, Global DOFS = 42875, Max Local DOFS = 314432, Degree = 1, Time taken = 0.0011335 s, DOFS/s = 3.78253e+07, DOFS/(s*rank) = 3.78253e+07
    Device Gather:        Ranks = 1, Global DOFS = 42875, Max Local DOFS = 314432, Degree = 1, Time taken = 1.45e-05 s, DOFS/s = 2.9569e+09, DOFS/(s*rank) = 2.9569e+09
    Host Scatter:         Ranks = 1, Global DOFS = 42875, Max Local DOFS = 314432, Degree = 1, Time taken = 0.0013037 s, DOFS/s = 3.28872e+07, DOFS/(s*rank) = 3.28872e+07
    Device Scatter:       Ranks = 1, Global DOFS = 42875, Max Local DOFS = 314432, Degree = 1, Time taken = 1.52e-05 s, DOFS/s = 2.82072e+09, DOFS/(s*rank) = 2.82072e+09
    ...

How to clean build objects
--------------------------

To clean the `dawgs` build objects:

    $ cd /path/to/dawgs/repo
    $ make realclean

Please invoke `make help` for more supported options.
