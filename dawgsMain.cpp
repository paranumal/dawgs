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

dfloat CheckCorrectness(const int N,
                        memory<dfloat> qtest,
                        memory<dfloat> qcheck,
                        const memory<hlong> ids,
                        comm_t comm) {

  for (dlong n=0;n<N;n++) {
    if (fabs(qtest[n]-qcheck[n])>1.0e-6) {
      printf("Rank %d, Entry %d, baseId %lld q = %f, qRef = %f \n", comm.rank(), n, ids[n], qtest[n], qcheck[n]);
    }
  }

  dfloat err=0.0;
  for (dlong n=0;n<N;n++) err += fabs(qtest[n]-qcheck[n]);

  comm.Allreduce(err);

  return err;
}

memory<hlong> MakeIds(dawgsSettings_t &settings) {

  comm_t& comm = settings.comm;

  //number of MPI ranks
  int size = comm.size();
  //global MPI rank
  int rank = comm.rank();

  // find a factorization size = size_x*size_y*size_z such that
  //  size_x>=size_y>=size_z are all 'close' to one another
  int size_x, size_y, size_z;
  Factor3(size, size_x, size_y, size_z);

  //determine (x,y,z) rank coordinates for this processes
  int rank_x=-1, rank_y=-1, rank_z=-1;
  RankDecomp3(size_x, size_y, size_z,
              rank_x, rank_y, rank_z,
              rank);

  //get polynomial degree
  int N;
  settings.getSetting("POLYNOMIAL DEGREE", N);

  //get global size from settings
  int NX, NY, NZ;
  settings.getSetting("BOX NX", NX);
  settings.getSetting("BOX NY", NY);
  settings.getSetting("BOX NZ", NZ);

  //get local size from settings
  int nx, ny, nz;
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

  dlong Nelements = nx*ny*nz;

  //find what global offsets my indices will start at
  hlong NX_offset = rank_x * (NX/size_x) + ((rank_x < (NX % size_x)) ? rank_x : (NX % size_x));
  hlong NY_offset = rank_y * (NY/size_y) + ((rank_y < (NY % size_y)) ? rank_y : (NY % size_y));
  hlong NZ_offset = rank_z * (NZ/size_z) + ((rank_z < (NZ % size_z)) ? rank_z : (NZ % size_z));

  int Nq = N+1; //number of points in 1D
  int Np = Nq*Nq*Nq; //number of points in full cube

  if (rank==0) {
    std::cout << "Ranks = " << size << ", ";
    std::cout << "Global DOFS = " << Np*NX*NY*NZ << ", ";
    std::cout << "Max Local DOFS = " << Np*Nelements << ", ";
    std::cout << "Degree = " << N << std::endl;
  }

  //Now make array of global indices mimiking a 3D box of cube elements
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

  return ids;
}


void PerformanceTest(platform_t &platform,
                     comm_t comm,
                     const int N,
                     memory<hlong> ids,
                     ogs::Method method) {

  int rank = comm.rank();
  int size = comm.size();

  int K = 1;

  dlong Ndofs = ids.length();

  //make an array
  memory<dfloat> q(K*Ndofs);

  /*Create rng*/
  std::mt19937 RNG = std::mt19937(comm.rank());
  std::uniform_real_distribution<> distrib(-0.25, 0.25);

  //fill with random numbers
  for (dlong n=0;n<K*Ndofs;n++) q[n]=distrib(RNG);

  //make a device array o_q, copying q from host on creation
  deviceMemory<dfloat> o_q = platform.malloc<dfloat>(K*Ndofs, q);

  bool verbose=true;
  bool unique = false;
  ogs::ogs_t ogs;
  ogs.Setup(Ndofs, ids, comm, ogs::Unsigned,
              method, unique, verbose, platform);

  unique = true;
  ogs::ogs_t sogs;
  sogs.Setup(Ndofs, ids, comm, ogs::Signed,
               method, unique, verbose, platform);

  deviceMemory<dfloat> o_gq = platform.malloc<dfloat>(K*sogs.Ngather);
  memory<dfloat> gq(K*sogs.Ngather);

  int Nwarmup = 10;
  int Ntests = 20;
  timePoint_t start, end;
  double elapsedTime;

  for (int n=0;n<Nwarmup;n++) {
    ogs.GatherScatter(q, K, ogs::Add, ogs::Sym);
  }
  start = GlobalPlatformTime(platform);
  for (int n=0;n<Ntests;n++) {
    ogs.GatherScatter(q, K, ogs::Add, ogs::Sym);
  }
  end = GlobalPlatformTime(platform);
  elapsedTime = ElapsedTime(start, end)/Ntests;

  if (rank==0) {
    std::cout << "Host GatherScatter:   Ranks = " << size << ", ";
    std::cout << "Global DOFS = " << ogs.NgatherGlobal << ", ";
    std::cout << "Max Local DOFS = " << Ndofs << ", ";
    std::cout << "Degree = " << N << ", ";
    std::cout << "Time taken = " << elapsedTime << " s, ";
    std::cout << "DOFS/s = " <<  (ogs.NgatherGlobal)/elapsedTime << ", ";
    std::cout << "DOFS/(s*rank) = " <<  (ogs.NgatherGlobal)/(elapsedTime*size) << std::endl;
  }

  for (int n=0;n<Nwarmup;n++) {
    ogs.GatherScatter(o_q, K, ogs::Add, ogs::Sym);
  }
  start = GlobalPlatformTime(platform);
  for (int n=0;n<Ntests;n++) {
    ogs.GatherScatter(o_q, K, ogs::Add, ogs::Sym);
  }
  end = GlobalPlatformTime(platform);
  elapsedTime = ElapsedTime(start, end)/Ntests;

  if (rank==0) {
    std::cout << "Device GatherScatter: Ranks = " << size << ", ";
    std::cout << "Global DOFS = " << ogs.NgatherGlobal << ", ";
    std::cout << "Max Local DOFS = " << Ndofs << ", ";
    std::cout << "Degree = " << N << ", ";
    std::cout << "Time taken = " << elapsedTime << " s, ";
    std::cout << "DOFS/s = " <<  (ogs.NgatherGlobal)/elapsedTime << ", ";
    std::cout << "DOFS/(s*rank) = " <<  (ogs.NgatherGlobal)/(elapsedTime*size) << std::endl;
  }


  for (int n=0;n<Nwarmup;n++) {
    sogs.Gather(gq, q, K, ogs::Add, ogs::Trans);
  }
  start = GlobalPlatformTime(platform);
  for (int n=0;n<Ntests;n++) {
    sogs.Gather(gq, q, K, ogs::Add, ogs::Trans);
  }
  end = GlobalPlatformTime(platform);
  elapsedTime = ElapsedTime(start, end)/Ntests;

  if (rank==0) {
    std::cout << "Host Gather:          Ranks = " << size << ", ";
    std::cout << "Global DOFS = " << ogs.NgatherGlobal << ", ";
    std::cout << "Max Local DOFS = " << Ndofs << ", ";
    std::cout << "Degree = " << N << ", ";
    std::cout << "Time taken = " << elapsedTime << " s, ";
    std::cout << "DOFS/s = " <<  (ogs.NgatherGlobal)/elapsedTime << ", ";
    std::cout << "DOFS/(s*rank) = " <<  (ogs.NgatherGlobal)/(elapsedTime*size) << std::endl;
  }

  for (int n=0;n<Nwarmup;n++) {
    sogs.Gather(o_gq, o_q, K, ogs::Add, ogs::Trans);
  }
  start = GlobalPlatformTime(platform);
  for (int n=0;n<Ntests;n++) {
    sogs.Gather(o_gq, o_q, K, ogs::Add, ogs::Trans);
  }
  end = GlobalPlatformTime(platform);
  elapsedTime = ElapsedTime(start, end)/Ntests;

  if (rank==0) {
    std::cout << "Device Gather:        Ranks = " << size << ", ";
    std::cout << "Global DOFS = " << ogs.NgatherGlobal << ", ";
    std::cout << "Max Local DOFS = " << Ndofs << ", ";
    std::cout << "Degree = " << N << ", ";
    std::cout << "Time taken = " << elapsedTime << " s, ";
    std::cout << "DOFS/s = " <<  (ogs.NgatherGlobal)/elapsedTime << ", ";
    std::cout << "DOFS/(s*rank) = " <<  (ogs.NgatherGlobal)/(elapsedTime*size) << std::endl;
  }

  for (int n=0;n<Nwarmup;n++) {
    sogs.Scatter(q, gq, K, ogs::NoTrans);
  }
  start = GlobalPlatformTime(platform);
  for (int n=0;n<Ntests;n++) {
    sogs.Scatter(q, gq, K, ogs::NoTrans);
  }
  end = GlobalPlatformTime(platform);
  elapsedTime = ElapsedTime(start, end)/Ntests;

  if (rank==0) {
    std::cout << "Host Scatter:         Ranks = " << size << ", ";
    std::cout << "Global DOFS = " << ogs.NgatherGlobal << ", ";
    std::cout << "Max Local DOFS = " << Ndofs << ", ";
    std::cout << "Degree = " << N << ", ";
    std::cout << "Time taken = " << elapsedTime << " s, ";
    std::cout << "DOFS/s = " <<  (ogs.NgatherGlobal)/elapsedTime << ", ";
    std::cout << "DOFS/(s*rank) = " <<  (ogs.NgatherGlobal)/(elapsedTime*size) << std::endl;
  }

  for (int n=0;n<Nwarmup;n++) {
    sogs.Scatter(o_q, o_gq, K, ogs::NoTrans);
  }
  start = GlobalPlatformTime(platform);
  for (int n=0;n<Ntests;n++) {
    sogs.Scatter(o_q, o_gq, K, ogs::NoTrans);
  }
  end = GlobalPlatformTime(platform);
  elapsedTime = ElapsedTime(start, end)/Ntests;

  if (rank==0) {
    std::cout << "Device Scatter:       Ranks = " << size << ", ";
    std::cout << "Global DOFS = " << ogs.NgatherGlobal << ", ";
    std::cout << "Max Local DOFS = " << Ndofs << ", ";
    std::cout << "Degree = " << N << ", ";
    std::cout << "Time taken = " << elapsedTime << " s, ";
    std::cout << "DOFS/s = " <<  (ogs.NgatherGlobal)/elapsedTime << ", ";
    std::cout << "DOFS/(s*rank) = " <<  (ogs.NgatherGlobal)/(elapsedTime*size) << std::endl;
  }
}


void CorrectnessTest(platform_t &platform,
                     comm_t comm,
                     memory<hlong> ids,
                     ogs::Method method) {
  int K = 1;

  dlong Ndofs = ids.length();

  //make an array
  memory<dfloat> q(K*Ndofs);

  /*Create rng*/
  std::mt19937 RNG = std::mt19937(comm.rank());
  std::uniform_real_distribution<> distrib(-0.25, 0.25);

  //fill with random numbers
  for (dlong n=0;n<K*Ndofs;n++) q[n]=distrib(RNG);

  //make a device array o_q, copying q from host on creation
  deviceMemory<dfloat> o_q = platform.malloc<dfloat>(K*Ndofs, q);

  //make a host gs handle (calls gslib)
  int verbose = 0;
  int iunique = 0;
  void *gsHandle = gsSetup(comm.comm(), Ndofs, ids.ptr(), iunique, verbose);

  ogs::ogs_t ogs;
  ogs::ogs_t sogs;

  //populate an array with the result we expect
  memory<dfloat> qcheck(K*Ndofs);

  deviceMemory<dfloat> o_gq;
  memory<dfloat> gq;

  //make the golden result
  int transpose = 0;
  qcheck.copyFrom(q);
  gsGatherScatterVec(qcheck.ptr(), K, gsHandle, transpose);

  dfloat err;
  memory<dfloat> qtest(K*Ndofs);

  verbose=true;
  bool unique = false;
  ogs.Setup(Ndofs, ids, comm, ogs::Unsigned,
              method, unique, verbose, platform);

  unique = true;
  sogs.Setup(Ndofs, ids, comm, ogs::Signed,
               method, unique, verbose, platform);

  o_gq = platform.malloc<dfloat>(K*sogs.Ngather);
  gq.malloc(K*sogs.Ngather);

  o_q.copyFrom(q);
  ogs.GatherScatter(o_q, K, ogs::Add, ogs::Sym);
  o_q.copyTo(qtest);
  err = CheckCorrectness(K*Ndofs, qtest, qcheck, ids, comm);
  if (comm.rank()==0) {
    std::cout << "Device GatherScatter:  Error = " << err << std::endl;
  }

  qtest.copyFrom(q);
  ogs.GatherScatter(qtest, K, ogs::Add, ogs::Sym);
  err = CheckCorrectness(K*Ndofs, qtest, qcheck, ids, comm);
  if (comm.rank()==0) {
    std::cout << "Host   GatherScatter:  Error = " << err << std::endl;
  }

  o_q.copyFrom(q);
  sogs.GatherScatter(o_q, K, ogs::Add, ogs::Sym);
  o_q.copyTo(qtest);
  err = CheckCorrectness(K*Ndofs, qtest, qcheck, ids, comm);
  if (comm.rank()==0) {
    std::cout << "Device GatherScatter:  Error = " << err << std::endl;
  }

  qtest.copyFrom(q);
  sogs.GatherScatter(qtest, K, ogs::Add, ogs::Sym);
  err = CheckCorrectness(K*Ndofs, qtest, qcheck, ids, comm);
  if (comm.rank()==0) {
    std::cout << "Host   GatherScatter:  Error = " << err << std::endl;
  }

  o_q.copyFrom(q);
  sogs.Gather (o_gq, o_q, K, ogs::Add, ogs::Trans);
  sogs.Scatter(o_q, o_gq, K, ogs::NoTrans);
  o_q.copyTo(qtest);
  err = CheckCorrectness(K*Ndofs, qtest, qcheck, ids, comm);
  if (comm.rank()==0) {
    std::cout << "Device Gather+Scatter: Error = " << err << std::endl;
  }

  sogs.Gather (gq, q, K, ogs::Add, ogs::Trans);
  sogs.Scatter(qtest, gq, K, ogs::NoTrans);
  err = CheckCorrectness(K*Ndofs, qtest, qcheck, ids, comm);
  if (comm.rank()==0) {
    std::cout << "Host   Gather+Scatter: Error = " << err << std::endl;
  }

  gsFree(gsHandle);
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

    if (settings.compareSetting("VERBOSE", "TRUE"))
      settings.report();

    bool sweep;
    sweep = settings.compareSetting("SWEEP", "TRUE");

    bool check;
    check = settings.compareSetting("CORRECTNESS CHECK", "TRUE");

    if (!sweep) {
      //get polynomial degree
      int N;
      settings.getSetting("POLYNOMIAL DEGREE", N);

      memory<hlong> ids;

      if (check) {
        if (settings.compareSetting("METHOD", "Pairwise")) {
          if (comm.rank()==0) std::cout << " ---- Pairwise ----" << std::endl;
          ids = MakeIds(settings);
          CorrectnessTest(platform, comm, ids, ogs::Pairwise);
        } else if (settings.compareSetting("METHOD", "Alltoall")) {
          if (comm.rank()==0) std::cout << " ---- Alltoall ----" << std::endl;
          ids = MakeIds(settings);
          CorrectnessTest(platform, comm, ids, ogs::AllToAll);
          if (comm.rank()==0) std::cout << " ---- CrystalRouter ----" << std::endl;
        } else if (settings.compareSetting("METHOD", "CrystalRouter")) {
          ids = MakeIds(settings);
          CorrectnessTest(platform, comm, ids, ogs::CrystalRouter);
        } else {
          if (comm.rank()==0) std::cout << " ---- Pairwise ----" << std::endl;
          ids = MakeIds(settings);
          CorrectnessTest(platform, comm, ids, ogs::Pairwise);
          if (comm.rank()==0) std::cout << " ---- Alltoall ----" << std::endl;
          ids = MakeIds(settings);
          CorrectnessTest(platform, comm, ids, ogs::AllToAll);
          if (comm.rank()==0) std::cout << " ---- CrystalRouter ----" << std::endl;
          ids = MakeIds(settings);
          CorrectnessTest(platform, comm, ids, ogs::CrystalRouter);
          if (comm.rank()==0) std::cout << " ---- Auto ----" << std::endl;
          ids = MakeIds(settings);
          CorrectnessTest(platform, comm, ids, ogs::Auto);
        }
      } else {
        if (settings.compareSetting("METHOD", "Pairwise")) {
          if (comm.rank()==0) std::cout << " ---- Pairwise ----" << std::endl;
          ids = MakeIds(settings);
          PerformanceTest(platform, comm, N, ids, ogs::Pairwise);
        } else if (settings.compareSetting("METHOD", "Alltoall")) {
          if (comm.rank()==0) std::cout << " ---- Alltoall ----" << std::endl;
          ids = MakeIds(settings);
          PerformanceTest(platform, comm, N, ids, ogs::AllToAll);
          if (comm.rank()==0) std::cout << " ---- CrystalRouter ----" << std::endl;
        } else if (settings.compareSetting("METHOD", "CrystalRouter")) {
          ids = MakeIds(settings);
          PerformanceTest(platform, comm, N, ids, ogs::CrystalRouter);
        } else {
          if (comm.rank()==0) std::cout << " ---- Pairwise ----" << std::endl;
          ids = MakeIds(settings);
          PerformanceTest(platform, comm, N, ids, ogs::Pairwise);
          if (comm.rank()==0) std::cout << " ---- Alltoall ----" << std::endl;
          ids = MakeIds(settings);
          PerformanceTest(platform, comm, N, ids, ogs::AllToAll);
          if (comm.rank()==0) std::cout << " ---- CrystalRouter ----" << std::endl;
          ids = MakeIds(settings);
          PerformanceTest(platform, comm, N, ids, ogs::CrystalRouter);
          if (comm.rank()==0) std::cout << " ---- Auto ----" << std::endl;
          ids = MakeIds(settings);
          PerformanceTest(platform, comm, N, ids, ogs::Auto);
        }
      }

    } else {
      //sweep through lots of tests
      std::vector<int> NN_low {  2,  2,  2,  2,  2,  2,  2,  2};
      std::vector<int> NN_high{122,102, 82, 62, 54, 38, 28, 28};
      std::vector<int> NN_step{  8,  4,  4,  4,  4,  2,  2,  2};

      settings.changeSetting("BOX NX", std::to_string(-1));
      settings.changeSetting("BOX NY", std::to_string(-1));
      settings.changeSetting("BOX NZ", std::to_string(-1));

      for (int N=1;N<9;N++) {

        const int low  = NN_low[N-1];
        const int high = NN_high[N-1];
        const int step = NN_step[N-1];

        for (int NN=low;NN<=high;NN+=step) {
          settings.changeSetting("LOCAL BOX NX", std::to_string(NN));
          settings.changeSetting("LOCAL BOX NY", std::to_string(NN));
          settings.changeSetting("LOCAL BOX NZ", std::to_string(NN));

          memory<hlong> ids;

          if (settings.compareSetting("METHOD", "Pairwise")) {
            if (comm.rank()==0) std::cout << " ---- Pairwise ----" << std::endl;
            ids = MakeIds(settings);
            PerformanceTest(platform, comm, N, ids, ogs::Pairwise);
          } else if (settings.compareSetting("METHOD", "Alltoall")) {
            if (comm.rank()==0) std::cout << " ---- Alltoall ----" << std::endl;
            ids = MakeIds(settings);
            PerformanceTest(platform, comm, N, ids, ogs::AllToAll);
            if (comm.rank()==0) std::cout << " ---- CrystalRouter ----" << std::endl;
          } else if (settings.compareSetting("METHOD", "CrystalRouter")) {
            ids = MakeIds(settings);
            PerformanceTest(platform, comm, N, ids, ogs::CrystalRouter);
          } else {
            if (comm.rank()==0) std::cout << " ---- Pairwise ----" << std::endl;
            ids = MakeIds(settings);
            PerformanceTest(platform, comm, N, ids, ogs::Pairwise);
            if (comm.rank()==0) std::cout << " ---- Alltoall ----" << std::endl;
            ids = MakeIds(settings);
            PerformanceTest(platform, comm, N, ids, ogs::AllToAll);
            if (comm.rank()==0) std::cout << " ---- CrystalRouter ----" << std::endl;
            ids = MakeIds(settings);
            PerformanceTest(platform, comm, N, ids, ogs::CrystalRouter);
            if (comm.rank()==0) std::cout << " ---- Auto ----" << std::endl;
            ids = MakeIds(settings);
            PerformanceTest(platform, comm, N, ids, ogs::Auto);
          }
        }
      }
    }
  }

  // close down MPI
  comm_t::Finalize();
  return LIBP_SUCCESS;
}

