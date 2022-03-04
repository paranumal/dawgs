/*
The MIT License (MIT)

Copyright (c) 2017-2021 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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
#include <random>

void CorrectnessTest(const int N,
                     memory<dfloat> qtest,
                     memory<dfloat> qcheck,
                     const memory<hlong> ids,
                     const std::string testName,
                     comm_t comm) {

  for (dlong n=0;n<N;n++) {
    if (fabs(qtest[n]-qcheck[n])>1.0e-6) {
      printf("Rank %d, Entry %d, baseId %lld q = %f, qRef = %f \n", comm.rank(), n, ids[n], qtest[n], qcheck[n]);
    }
  }

  dfloat err=0.0;
  for (dlong n=0;n<N;n++) err += fabs(qtest[n]-qcheck[n]);

  comm.Reduce(err, 0);

  if (comm.rank()==0) {
    std::cout << testName << ": Error = " << err << std::endl;
  }
}

/*
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
    if (Nvectors) o_a.copyTo(o_b, Nlocal*Nvectors*sizeof(dfloat), 0, 0, "async: true");
    ogs.GatherScatterFinish(o_q, method, gpu_aware, overlap);
    ogs.platform.device.finish();
  }

  int n_iter = 50;
  dfloat starttime, endtime;

  // std::vector<int> Nvec{0, 1, 3, 7};
  std::vector<int> Nvec{0};

  for (int m : Nvec) {
    MPI_Barrier(comm);
    starttime = MPI_Wtime();

    for (int n=0;n<n_iter;n++) {
      ogs.GatherScatterStart(o_q, method, gpu_aware, overlap);
      if (Nvectors) o_a.copyTo(o_b, Nlocal*m*sizeof(dfloat), 0, 0, "async: true");
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
*/

void Test(platform_t & platform, comm_t comm, dawgsSettings_t& settings,
          const dlong nx, const dlong ny, const dlong nz,
          const dlong NX, const dlong NY, const dlong NZ,
          const int N) {

  //number of MPI ranks
  int size = platform.size();
  //global MPI rank
  int rank = platform.rank();

  // find a factorization size = size_x*size_y*size_z such that
  //  size_x>=size_y>=size_z are all 'close' to one another
  int size_x, size_y, size_z;
  Factor3(size, size_x, size_y, size_z);

  //determine (x,y,z) rank coordinates for this processes
  int rank_x=-1, rank_y=-1, rank_z=-1;
  RankDecomp3(size_x, size_y, size_z,
              rank_x, rank_y, rank_z,
              rank);

  if (settings.compareSetting("VERBOSE", "TRUE"))
    settings.report();

  if (rank==0 && settings.compareSetting("VERBOSE", "TRUE")) {
    std::cout << "MPI grid configuration: " << size_x << " x "
                                            << size_y << " x "
                                            << size_z << std::endl;
  }

  dlong Nelements = nx*ny*nz;

  //find what global offsets my indices will start at
  hlong NX_offset = rank_x * (NX/size_x) + ((rank_x < (NX % size_x)) ? rank_x : (NX % size_x));
  hlong NY_offset = rank_y * (NY/size_y) + ((rank_y < (NY % size_y)) ? rank_y : (NY % size_y));
  hlong NZ_offset = rank_z * (NZ/size_z) + ((rank_z < (NZ % size_z)) ? rank_z : (NZ % size_z));

  int Nq = N+1; //number of points in 1D
  int Np = Nq*Nq*Nq; //number of points in full cube

  //Now make array of global indices mimiking a 3D box of cube elements

  // hlong is usually a 64-bit integer type
  memory<hlong> ids(Nelements*Np);

  for (int K=0;K<nz;K++) {
    for (int J=0;J<ny;J++) {
      for (int I=0;I<nx;I++) {

        hlong *ids_e = ids.ptr() + (I + J*nx + K*nx*ny)*Np;

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

  int K = 1;
  //make an array
  memory<dfloat> q(K*Nelements*Np);

  /*Create rng*/
  std::mt19937 RNG = std::mt19937(rank);
  std::uniform_real_distribution<> distrib(-0.25, 0.25);

  //fill with random numbers
  for (dlong n=0;n<K*Nelements*Np;n++) q[n]=distrib(RNG);

  //make a device array o_q, copying q from host on creation
  deviceMemory<dfloat> o_q = platform.malloc<dfloat>(K*Nelements*Np, q);

  if (rank==0) {
    std::cout << "Ranks = " << size << ", ";
    std::cout << "Global DOFS = " << Np*NX*NY*NZ << ", ";
    std::cout << "Max Local DOFS = " << Np*Nelements << ", ";
    std::cout << "Degree = " << N << std::endl;
  }


  // if (settings.compareSetting("CORRECTNESS CHECK", "TRUE")) {
    /*************************
     * Test correctness
     *************************/

    //make a host gs handle (calls gslib)
    int verbose = 0;
    int iunique = 0;
    void *gsHandle = gsSetup(comm.comm(), Nelements*Np, ids.ptr(), iunique, verbose);

    ogs::ogs_t ogs, p_ogs, a_ogs, c_ogs;
    ogs::ogs_t sogs, p_sogs, a_sogs, c_sogs;

    //populate an array with the result we expect
    memory<dfloat> qcheck(K*Nelements*Np);

    deviceMemory<dfloat> o_gq;
    memory<dfloat> gq;

    //make the golden result
    int transpose = 0;
    qcheck.copyFrom(q);
    gsGatherScatterVec(qcheck.ptr(), K, gsHandle, transpose);


    memory<dfloat> qtest(K*Nelements*Np);

    if (comm.rank()==0)
      std::cout << "---------- Pairwise ---------" << std::endl;

    bool unique = false;
    p_ogs.Setup(Nelements*Np, ids, comm, ogs::Signed,
                ogs::Pairwise, unique, verbose, platform);

    unique = true;
    p_sogs.Setup(Nelements*Np, ids, comm, ogs::Signed,
                 ogs::Pairwise, unique, verbose, platform);

    o_gq = platform.malloc<dfloat>(K*p_sogs.Ngather);
    gq.malloc(K*p_sogs.Ngather);

    o_q.copyFrom(q);
    p_ogs.GatherScatter(o_q, K, ogs::Add, ogs::Sym);
    o_q.copyTo(qtest);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Device GatherScatter", comm);

    qtest.copyFrom(q);
    p_ogs.GatherScatter(qtest, K, ogs::Add, ogs::Sym);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Host GatherScatter", comm);


    o_q.copyFrom(q);
    p_sogs.GatherScatter(o_q, K, ogs::Add, ogs::Sym);
    o_q.copyTo(qtest);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Device GatherScatter", comm);

    qtest.copyFrom(q);
    p_sogs.GatherScatter(qtest, K, ogs::Add, ogs::Sym);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Host GatherScatter", comm);

    o_q.copyFrom(q);
    p_sogs.Gather (o_gq, o_q, K, ogs::Add, ogs::Trans);
    p_sogs.Scatter(o_q, o_gq, K, ogs::NoTrans);
    o_q.copyTo(qtest);

    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Device Gather+Scatter", comm);

    p_sogs.Gather (gq, q, K, ogs::Add, ogs::Trans);
    p_sogs.Scatter(qtest, gq, K, ogs::NoTrans);

    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Host Gather+Scatter", comm);

    p_ogs.Free();
    p_sogs.Free();
    gq.free();
    o_gq.free();

    if (comm.rank()==0)
      std::cout << "---------- All-to-all ---------" << std::endl;

    unique = false;
    a_ogs.Setup(Nelements*Np, ids, comm, ogs::Signed,
                ogs::AllToAll, unique, verbose, platform);

    unique = true;
    a_sogs.Setup(Nelements*Np, ids, comm, ogs::Signed,
                 ogs::AllToAll, unique, verbose, platform);

    o_gq = platform.malloc<dfloat>(K*a_sogs.Ngather);
    gq.malloc(K*a_sogs.Ngather);

    o_q.copyFrom(q);
    a_ogs.GatherScatter(o_q, K, ogs::Add, ogs::Sym);
    o_q.copyTo(qtest);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Device GatherScatter", comm);

    qtest.copyFrom(q);
    a_ogs.GatherScatter(qtest, K, ogs::Add, ogs::Sym);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Host GatherScatter", comm);


    o_q.copyFrom(q);
    a_sogs.GatherScatter(o_q, K, ogs::Add, ogs::Sym);
    o_q.copyTo(qtest);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Device GatherScatter", comm);

    qtest.copyFrom(q);
    a_sogs.GatherScatter(qtest, K, ogs::Add, ogs::Sym);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Host GatherScatter", comm);

    o_q.copyFrom(q);
    a_sogs.Gather (o_gq, o_q, K, ogs::Add, ogs::Trans);
    a_sogs.Scatter(o_q, o_gq, K, ogs::NoTrans);
    o_q.copyTo(qtest);

    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Device Gather+Scatter", comm);

    a_sogs.Gather (gq, q, K, ogs::Add, ogs::Trans);
    a_sogs.Scatter(qtest, gq, K, ogs::NoTrans);

    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Host Gather+Scatter", comm);

    a_ogs.Free();
    a_sogs.Free();
    gq.free();
    o_gq.free();

    if (comm.rank()==0)
      std::cout << "---------- Crystal Router ---------" << std::endl;

    unique = false;
    c_ogs.Setup(Nelements*Np, ids, comm, ogs::Signed,
                ogs::CrystalRouter, unique, verbose, platform);

    unique = true;
    c_sogs.Setup(Nelements*Np, ids, comm, ogs::Signed,
                 ogs::CrystalRouter, unique, verbose, platform);

    o_gq = platform.malloc<dfloat>(K*c_sogs.Ngather);
    gq.malloc(K*c_sogs.Ngather);

    o_q.copyFrom(q);
    c_ogs.GatherScatter(o_q, K, ogs::Add, ogs::Sym);
    o_q.copyTo(qtest);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Device GatherScatter", comm);

    qtest.copyFrom(q);
    c_ogs.GatherScatter(qtest, K, ogs::Add, ogs::Sym);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Host GatherScatter", comm);


    o_q.copyFrom(q);
    c_sogs.GatherScatter(o_q, K, ogs::Add, ogs::Sym);
    o_q.copyTo(qtest);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Device GatherScatter", comm);

    qtest.copyFrom(q);
    c_sogs.GatherScatter(qtest, K, ogs::Add, ogs::Sym);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Host GatherScatter", comm);

    o_q.copyFrom(q);
    c_sogs.Gather (o_gq, o_q, K, ogs::Add, ogs::Trans);
    c_sogs.Scatter(o_q, o_gq, K, ogs::NoTrans);
    o_q.copyTo(qtest);

    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Device Gather+Scatter", comm);

    c_sogs.Gather (gq, q, K, ogs::Add, ogs::Trans);
    c_sogs.Scatter(qtest, gq, K, ogs::NoTrans);

    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Host Gather+Scatter", comm);

    c_ogs.Free();
    c_sogs.Free();
    gq.free();
    o_gq.free();

    if (comm.rank()==0)
      std::cout << "---------- Auto ---------" << std::endl;

    verbose=true;
    unique = false;
    ogs.Setup(Nelements*Np, ids, comm, ogs::Signed,
                ogs::Auto, unique, verbose, platform);

    unique = true;
    sogs.Setup(Nelements*Np, ids, comm, ogs::Signed,
                 ogs::Auto, unique, verbose, platform);

    o_gq = platform.malloc<dfloat>(K*sogs.Ngather);
    gq.malloc(K*sogs.Ngather);

    o_q.copyFrom(q);
    ogs.GatherScatter(o_q, K, ogs::Add, ogs::Sym);
    o_q.copyTo(qtest);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Device GatherScatter", comm);

    qtest.copyFrom(q);
    ogs.GatherScatter(qtest, K, ogs::Add, ogs::Sym);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Host GatherScatter", comm);


    o_q.copyFrom(q);
    sogs.GatherScatter(o_q, K, ogs::Add, ogs::Sym);
    o_q.copyTo(qtest);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Device GatherScatter", comm);

    qtest.copyFrom(q);
    sogs.GatherScatter(qtest, K, ogs::Add, ogs::Sym);
    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Host GatherScatter", comm);

    o_q.copyFrom(q);
    sogs.Gather (o_gq, o_q, K, ogs::Add, ogs::Trans);
    sogs.Scatter(o_q, o_gq, K, ogs::NoTrans);
    o_q.copyTo(qtest);

    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Device Gather+Scatter", comm);

    sogs.Gather (gq, q, K, ogs::Add, ogs::Trans);
    sogs.Scatter(qtest, gq, K, ogs::NoTrans);

    CorrectnessTest(K*Nelements*Np, qtest, qcheck, ids,
                    "Host Gather+Scatter", comm);

    ogs.Free();
    sogs.Free();
    gq.free();
    o_gq.free();

    // memory<dlong> GlobalToLocal(Nelements*Np);
    // ogs.SetupGlobalToLocalMapping(GlobalToLocal);

#if 0
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
#endif
}

int main(int argc, char **argv){

  // start up MPI
  comm_t::Init(argc, argv);

  {
    comm_t comm(comm_t::world().Dup());

    //parse run settings from cmd line
    dawgsSettings_t settings(argc, argv, comm);

    // set up platform (wraps OCCA device)
    platform_t platform(settings);

    //Trigger JIT kernel builds
    ogs::InitializeKernels(platform, ogs::Dfloat, ogs::Add);

    /*************************
     * Setup
     *************************/
    //number of MPI ranks
    int size = platform.size();
    //global MPI rank
    int rank = platform.rank();

    // find a factorization size = size_x*size_y*size_z such that
    //  size_x>=size_y>=size_z are all 'close' to one another
    int size_x, size_y, size_z;
    Factor3(size, size_x, size_y, size_z);

    //determine (x,y,z) rank coordinates for this processes
    int rank_x=-1, rank_y=-1, rank_z=-1;
    RankDecomp3(size_x, size_y, size_z,
                rank_x, rank_y, rank_z,
                rank);

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

      Test(platform, comm, settings, nx, ny, nz, NX, NY, NZ, N);
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
  }

  // close down MPI
  comm_t::Finalize();
  return LIBP_SUCCESS;
}

