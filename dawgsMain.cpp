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

#include "dawgs.hpp"

int Nvectors;
occa::memory o_a, o_b;

// find a factorization size = size_x*size_y*size_z such that
//  size_x>=size_y>=size_z are all 'close' to one another
void factor3(const int size, int &size_x, int &size_y, int &size_z) {
  //start with guessing size_x ~= size^1/3
  size_x = round(std::cbrt(size));
  size_y = size_z = 1;

  for (;size_x<size;size_x++) {
    if (size % size_x ==0) { //if size_x divides size
      int f = size / size_x; //divide out size_x

      size_y = round(sqrt(f)); //guess size_y ~= sqrt(f)
      for (;size_y<f;size_y++) {
        if (f % size_y == 0) { //if size_y divides f
          size_z = f/size_y; //divide out size_y

          //swap if needed
          if (size_y>size_x) {std::swap(size_x,size_y);}
          if (size_z>size_y) {std::swap(size_y,size_z);}
          if (size_y>size_x) {std::swap(size_x,size_y);}

          return;
        }
      }

      //if we're here, f is prime
      size_y = f;
      size_z = 1;

      //swap if needed
      if (size_y>size_x) {int tmp=size_x; size_x=size_y; size_y=tmp;}

      return;
    }
  }

  //if we made it this far, size is prime
  size_x = size;
  size_y = size_z = 1;
}

void CorrectnessTest(const int N, dfloat *q, occa::memory &o_q,
                     dfloat *qtest, dfloat *qcheck, hlong* ids,
                     ogs::ogs_t &ogs, const ogs::ogs_method method,
                     bool gpu_aware, bool overlap, MPI_Comm comm) {
  int rank = ogs.platform.rank;

  o_q.copyFrom(q);

  //call a gatherScatter operation
  ogs.GatherScatter(o_q, method, gpu_aware, overlap);

  //copy back to host
  o_q.copyTo(qtest);

  for (dlong n=0;n<N;n++) {
    if (fabs(qtest[n]-qcheck[n])>0.0) {
      printf("Rank %d, Entry %d, baseId %lld Error= %f \n", rank, n, ids[n], fabs(qtest[n]-qcheck[n]));
    }
  }

  dfloat err=0.0;
  for (dlong n=0;n<N;n++) err += fabs(qtest[n]-qcheck[n]);

  dfloat errG=0.0;
  MPI_Reduce(&err, &errG, 1, MPI_DFLOAT, MPI_SUM, 0, comm);

  if (rank==0) {
    if (method==ogs::ogs_all_reduce)
      std::cout << "AllToAll Method ";
    else if (method==ogs::ogs_pairwise)
      std::cout << "Pairwise Method ";
    else
      std::cout << "Crystal Router Method ";

    if (gpu_aware)
      std::cout << ", GPU-aware ";

    if (overlap)
      std::cout << ", Halo Kernel Overlap ";

    std::cout << ", Error = " << errG << std::endl;
  }
}


void PerformanceTest(int N, int64_t Ndofs, int Nlocal,
                     occa::memory &o_q, ogs::ogs_t &ogs,
                     const ogs::ogs_method method,
                     bool gpu_aware, bool overlap, MPI_Comm comm) {

  int rank = ogs.platform.rank;
  int size = ogs.platform.size;

  int Nwarmup = 10;
  MPI_Barrier(comm);
  for (int n=0;n<Nwarmup;n++) {
    ogs.GatherScatterStart(o_q, method, gpu_aware, overlap);
    //o_a.copyTo(o_b, o_a.size(), 0, 0, "async: true");
    ogs.GatherScatterFinish(o_q, method, gpu_aware, overlap);
    ogs.platform.device.finish();
  }

  int n_iter = 50;
  dfloat starttime, endtime;

  std::vector<int> Nvec{0, 1, 3, 7};

  for (int m : Nvec) {
    MPI_Barrier(comm);
    starttime = MPI_Wtime();

    for (int n=0;n<n_iter;n++) {
      ogs.GatherScatterStart(o_q, method, gpu_aware, overlap);
      o_a.copyTo(o_b, Nlocal*m*sizeof(dfloat), 0, 0, "async: true");
      ogs.GatherScatterFinish(o_q, method, gpu_aware, overlap);
      ogs.platform.device.finish();
    }

    //platform.device.finish();
    MPI_Barrier(comm);
    endtime = MPI_Wtime();

    double elapsed = (endtime-starttime)*1000/n_iter;

    if (rank==0) {
      if (method==ogs::ogs_all_reduce)
        std::cout << "AR ";
      else if (method==ogs::ogs_pairwise)
        std::cout << "PW ";
      else
        std::cout << "CR ";

      if (gpu_aware)
        std::cout << ", GPU-aware ";

      if (overlap)
        std::cout << ", Overlap ";

      std::cout << ": Ranks = " << size << ", ";
      std::cout << "Global DOFS = " << Ndofs << ", ";
      std::cout << "Max Local DOFS = " << Nlocal << ", ";
      std::cout << "Degree = " << N << ", ";
      std::cout << "Nvectors = " << m << ", ";
      std::cout << "Time taken = " << elapsed << " ms, ";
      std::cout << "DOFS/s = " <<  ((1+m)*Ndofs*1000.0)/elapsed << ", ";
      std::cout << "DOFS/(s*rank) = " <<  ((1+m)*Ndofs*1000.0)/(elapsed*size) << std::endl;
    }
  }
}

void Test(platform_t & platform, MPI_Comm comm, dawgsSettings_t& settings,
          const dlong nx, const dlong ny, const dlong nz,
          const dlong NX, const dlong NY, const dlong NZ,
          const int N) {

  //number of MPI ranks
  int size = platform.size;

  // find a factorization size = size_x*size_y*size_z such that
  //  size_x>=size_y>=size_z are all 'close' to one another
  int size_x, size_y, size_z;
  factor3(size, size_x, size_y, size_z);

  //global MPI rank
  int rank = platform.rank;

  //find our coordinates in the MPI grid such that
  // rank = rank_x + rank_y*size_x + rank_z*size_x*size_y
  int rank_z = rank/(size_x*size_y);
  int rank_y = (rank-rank_z*size_x*size_y)/size_x;
  int rank_x = rank % size_x;

  //parse GPU-aware setting from cmd line
  bool gpu_aware;
  gpu_aware = settings.compareSetting("GPU AWARE", "TRUE");

  if (settings.compareSetting("VERBOSE", "TRUE"))
    settings.report();

  if (rank==0 && settings.compareSetting("VERBOSE", "TRUE")) {
    std::cout << "MPI grid configuration: " << size_x << " x "
                                            << size_y << " x "
                                            << size_z << std::endl;
  }

  dlong Nelements = nx*ny*nz;

  //find what global offsets my indices will start at
  dlong NX_offset = rank_x * (NX/size_x) + ((rank_x < (NX % size_x)) ? rank_x : (NX % size_x));
  dlong NY_offset = rank_y * (NY/size_y) + ((rank_y < (NY % size_y)) ? rank_y : (NY % size_y));
  dlong NZ_offset = rank_z * (NZ/size_z) + ((rank_z < (NZ % size_z)) ? rank_z : (NZ % size_z));

  int Nq = N+1; //number of points in 1D
  int Np = Nq*Nq*Nq; //number of points in full cube

  Nvectors=7;
  dfloat *a = (dfloat *) malloc(Nvectors*Nelements*Np*sizeof(dfloat));
  o_a = platform.malloc(Nvectors*Nelements*Np*sizeof(dfloat), a);
  o_b = platform.malloc(Nvectors*Nelements*Np*sizeof(dfloat), a);
  free(a);

  //Now make array of global indices mimiking a 3D box of cube elements

  // hlong is usually a 64-bit integer type
  hlong *ids = (hlong *) malloc(Nelements*Np*sizeof(hlong));

  for (int K=0;K<nz;K++) {
    for (int J=0;J<ny;J++) {
      for (int I=0;I<nx;I++) {

        hlong *ids_e = ids + (I + J*nx + K*nx*ny)*Np;

        hlong baseId =  (I + NX_offset)*N
                      + (J + NY_offset)*N*(N*NX+1)
                      + (K + NZ_offset)*N*(N*NX+1)*(N*NY+1)
                      + 1; //0 indcies are ignored, so shift everything by 1

        for (int k=0;k<Nq;k++) {
          for (int j=0;j<Nq;j++) {
            for (int i=0;i<Nq;i++) {
              ids_e[i+j*Nq+k*Nq*Nq] = i + j*(N*NX+1) + k*(N*NX+1)*(N*NY+1) + baseId;
            }
          }
        }
      }
    }
  }

  ogs::ogs_t ogs(platform);

  int verbose = 1;
  ogs.Setup(Nelements*Np, ids, comm, verbose);

  //make an array
  dfloat *q = (dfloat *) malloc(Nelements*Np*sizeof(dfloat));

  //fill with ones
  for (dlong n=0;n<Nelements*Np;n++) q[n]=1.0;

  //make a device array o_q, copying q from host on creation
  occa::memory o_q = platform.malloc(Nelements*Np*sizeof(dfloat), q);


  if (settings.compareSetting("CORRECTNESS CHECK", "TRUE")) {
    /*************************
     * Test correctness
     *************************/
    //make a host gs handle (calls gslib)
    void *gsHandle = gsSetup(comm, Nelements*Np, ids, 0, 0);

    if (rank==0) {
      std::cout << "Ranks = " << size << ", ";
      std::cout << "Global DOFS = " << Np*NX*NY*NZ << ", ";
      std::cout << "Max Local DOFS = " << Np*Nelements << ", ";
      std::cout << "Degree = " << N << std::endl;
    }

    //populate an array with the result we expect
    dfloat *qcheck = (dfloat *) malloc(Nelements*Np*sizeof(dfloat));

    for (dlong n=0;n<Nelements*Np;n++) qcheck[n] = q[n];

    //make the golden result
    gsGatherScatter(qcheck, gsHandle);

    dfloat *qtest = (dfloat *) malloc(Nelements*Np*sizeof(dfloat));

    CorrectnessTest(Nelements*Np, q, o_q,
                    qtest, qcheck, ids,
                    ogs, ogs::ogs_all_reduce, false, false, comm);

    CorrectnessTest(Nelements*Np, q, o_q,
                    qtest, qcheck, ids,
                    ogs, ogs::ogs_pairwise, false, false, comm);

    CorrectnessTest(Nelements*Np, q, o_q,
                    qtest, qcheck, ids,
                    ogs, ogs::ogs_crystal_router, false, false, comm);

    CorrectnessTest(Nelements*Np, q, o_q,
                    qtest, qcheck, ids,
                    ogs, ogs::ogs_all_reduce, false, true, comm);

    CorrectnessTest(Nelements*Np, q, o_q,
                    qtest, qcheck, ids,
                    ogs, ogs::ogs_pairwise, false, true, comm);

    CorrectnessTest(Nelements*Np, q, o_q,
                    qtest, qcheck, ids,
                    ogs, ogs::ogs_crystal_router, false, true, comm);

    if (gpu_aware) {
      CorrectnessTest(Nelements*Np, q, o_q,
                    qtest, qcheck, ids,
                    ogs, ogs::ogs_all_reduce, true, false, comm);

      CorrectnessTest(Nelements*Np, q, o_q,
                      qtest, qcheck, ids,
                      ogs, ogs::ogs_pairwise, true, false, comm);

      CorrectnessTest(Nelements*Np, q, o_q,
                      qtest, qcheck, ids,
                      ogs, ogs::ogs_crystal_router, true, false, comm);

      CorrectnessTest(Nelements*Np, q, o_q,
                    qtest, qcheck, ids,
                    ogs, ogs::ogs_all_reduce, true, true, comm);

      CorrectnessTest(Nelements*Np, q, o_q,
                      qtest, qcheck, ids,
                      ogs, ogs::ogs_pairwise, true, true, comm);

      CorrectnessTest(Nelements*Np, q, o_q,
                      qtest, qcheck, ids,
                      ogs, ogs::ogs_crystal_router, true, true, comm);
    }

  } else {
    /*************************
     * Performance Test
     *************************/
    int64_t Ndofs = ((int64_t) Np)*NX*NY*NZ;
    int Nlocal = Np*Nelements;

    //All to all
    PerformanceTest(N, Ndofs, Nlocal, o_q, ogs, ogs::ogs_all_reduce, false, false, comm);

    //Pairwise
    PerformanceTest(N, Ndofs, Nlocal, o_q, ogs, ogs::ogs_pairwise, false, false, comm);

    //Crystal Router
    PerformanceTest(N, Ndofs, Nlocal, o_q, ogs, ogs::ogs_crystal_router, false, false, comm);

    //With Halo kernel overlap:

    //All to all
    PerformanceTest(N, Ndofs, Nlocal, o_q, ogs, ogs::ogs_all_reduce, false, true, comm);

    //Pairwise
    PerformanceTest(N, Ndofs, Nlocal, o_q, ogs, ogs::ogs_pairwise, false, true, comm);

    //Crystal Router
    PerformanceTest(N, Ndofs, Nlocal, o_q, ogs, ogs::ogs_crystal_router, false, true, comm);

    if (gpu_aware) {
      //All to all
      PerformanceTest(N, Ndofs, Nlocal, o_q, ogs, ogs::ogs_all_reduce, true, false, comm);

      //Pairwise
      PerformanceTest(N, Ndofs, Nlocal, o_q, ogs, ogs::ogs_pairwise, true, false, comm);

      //Crystal Router
      PerformanceTest(N, Ndofs, Nlocal, o_q, ogs, ogs::ogs_crystal_router, true, false, comm);

      //With Halo kernel overlap:

      //All to all
      PerformanceTest(N, Ndofs, Nlocal, o_q, ogs, ogs::ogs_all_reduce, true, true, comm);

      //Pairwise
      PerformanceTest(N, Ndofs, Nlocal, o_q, ogs, ogs::ogs_pairwise, true, true, comm);

      //Crystal Router
      PerformanceTest(N, Ndofs, Nlocal, o_q, ogs, ogs::ogs_crystal_router, true, true, comm);
    }
  }

  free(ids);
}

int main(int argc, char **argv){

  // start up MPI
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  //parse run settings from cmd line
  dawgsSettings_t settings(argc, argv, comm);

  // set up platform (wraps OCCA device)
  platform_t platform(settings);

  //mkae an empty ogs object to trigger JIT kernel builds
  ogs::ogs_t ogs0(platform);

  /*************************
   * Setup
   *************************/
  //number of MPI ranks
  int size = platform.size;

  // find a factorization size = size_x*size_y*size_z such that
  //  size_x>=size_y>=size_z are all 'close' to one another
  int size_x, size_y, size_z;
  factor3(size, size_x, size_y, size_z);

  //global MPI rank
  int rank = platform.rank;

  //find our coordinates in the MPI grid such that
  // rank = rank_x + rank_y*size_x + rank_z*size_x*size_y
  int rank_z = rank/(size_x*size_y);
  int rank_y = (rank-rank_z*size_x*size_y)/size_x;
  int rank_x = rank % size_x;

  //number of cubes in each dimension
  dlong NX, NY, NZ; //global
  dlong nx, ny, nz; //local

  bool sweep;
  sweep = settings.compareSetting("SWEEP", "TRUE");

  if (!sweep) {
    //get polynomial degree
    int N;
    settings.getSetting("POLYNOMIAL DEGREE", N);

    //get global size from settings
    settings.getSetting("BOX NX", NX);
    settings.getSetting("BOX NY", NY);
    settings.getSetting("BOX NZ", NZ);

    //get local size from settings
    settings.getSetting("LOCAL BOX NX", nx);
    settings.getSetting("LOCAL BOX NY", ny);
    settings.getSetting("LOCAL BOX NZ", nz);

    if (NX*NY*NZ <= 0) { //if the user hasn't given global sizes
      //set global size by multiplying local size by grid dims
      NX = nx * size_x;
      NY = ny * size_y;
      NZ = nz * size_z;
      settings.changeSetting("BOX NX", std::to_string(NX));
      settings.changeSetting("BOX NY", std::to_string(NY));
      settings.changeSetting("BOX NZ", std::to_string(NZ));
    } else {
      //WARNING setting global sizes on input overrides any local sizes provided
      nx = NX/size_x + ((rank_x < (NX % size_x)) ? 1 : 0);
      ny = NY/size_y + ((rank_y < (NY % size_y)) ? 1 : 0);
      nz = NZ/size_z + ((rank_z < (NZ % size_z)) ? 1 : 0);
    }

    Test(platform, comm, settings, nx, ny, nz, NZ, NY, NZ, N);
  } else {
    //sweep through lots of tests
    std::vector<int> NN_low {  2,  2,  2,  2,  2,  2,  2,  2};
    std::vector<int> NN_high{122,102, 82, 62, 54, 38, 28, 28};
    std::vector<int> NN_step{  8,  4,  4,  4,  4,  2,  2,  2};

    for (int N=1;N<9;N++) {

      const int low  = NN_low[N-1];
      const int high = NN_high[N-1];
      const int step = NN_step[N-1];

      for (int NN=low;NN<=high;NN+=step) {
        nx = NN;
        ny = NN;
        nz = NN;
        NX = nx * size_x;
        NY = ny * size_y;
        NZ = nz * size_z;
        settings.changeSetting("BOX NX", std::to_string(NX));
        settings.changeSetting("BOX NY", std::to_string(NY));
        settings.changeSetting("BOX NZ", std::to_string(NZ));

        Test(platform, comm, settings, nx, ny, nz, NX, NY, NZ, N);
      }
    }
  }

  // close down MPI
  MPI_Finalize();
  return LIBP_SUCCESS;
}

