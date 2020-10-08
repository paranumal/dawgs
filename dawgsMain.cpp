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
          if (size_y>size_x) {int tmp=size_x; size_x=size_y; size_y=tmp;}
          if (size_z>size_y) {int tmp=size_y; size_y=size_z; size_z=tmp;}

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

int main(int argc, char **argv){

  // start up MPI
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  //parse run settings from cmd line
  dawgsSettings_t settings(argc, argv, comm);

  if (settings.compareSetting("GPU AWARE", "TRUE")
    &&(settings.compareSetting("THREAD MODEL","HIP"))) {
      settings.changeSetting("GPU AWARE", "TRUE");
    } else {
      settings.changeSetting("GPU AWARE", "FALSE");
    }
  
  if (settings.compareSetting("VERBOSE", "TRUE"))
    settings.report();

  // set up platform (wraps OCCA device)
  platform_t platform(settings);

  /*************************
   * Setup
   *************************/
  //number of MPI ranks
  int size = platform.size;

  // find a factorization size = size_x*size_y*size_z such that
  //  size_x>=size_y>=size_z are all 'close' to one another
  int size_x, size_y, size_z;
  factor3(size, size_x, size_y, size_z);

  //parse GPU-aware setting from cmd line
  bool gpu_aware;
  gpu_aware = settings.compareSetting("GPU AWARE", "TRUE");

  //global MPI rank
  int rank = platform.rank;

  if (rank==0) {
    std::cout << "Name:     [GPU AWARE]" << std::endl;
    std::cout << "CL keys:  [-ga, --gpu-aware]" << std::endl;
    if (gpu_aware)
      std::cout << "Value:    TRUE" << std::endl << std::endl;
    else
      std::cout << "Value:    FALSE" << std::endl << std::endl;
    
    std::cout << "MPI grid configuration: " << size_x << " x "
                                            << size_y << " x "
                                            << size_z << std::endl;
  }

  //find our coordinates in the MPI grid such that
  // rank = rank_x + rank_y*size_x + rank_z*size_x*size_y
  int rank_z = rank/(size_x*size_y);
  int rank_y = (rank-rank_z*size_x*size_y)/size_x;
  int rank_x = rank % size_x;

  //number of cubes in each dimension
  dlong NX, NY, NZ;
  settings.getSetting("BOX NX", NX);
  settings.getSetting("BOX NY", NY);
  settings.getSetting("BOX NZ", NZ);

  //compute number of cubes on my rank (adding 1 for remainders on some ranks)
  dlong nx = NX/size_x + ((rank_x < (NX % size_x)) ? 1 : 0);
  dlong ny = NY/size_y + ((rank_y < (NY % size_y)) ? 1 : 0);
  dlong nz = NZ/size_z + ((rank_z < (NZ % size_z)) ? 1 : 0);

  dlong Nelements = nx*ny*nz;

  //find what global offsets my indices will start at
  dlong NX_offset = rank_x * (NX/size_x) + ((rank_x < (NX % size_x)) ? rank_x : (NX % size_x));
  dlong NY_offset = rank_y * (NY/size_y) + ((rank_y < (NY % size_y)) ? rank_y : (NY % size_y));
  dlong NZ_offset = rank_z * (NZ/size_z) + ((rank_z < (NZ % size_z)) ? rank_z : (NZ % size_z));

  //get polynomial degree
  int N;
  settings.getSetting("POLYNOMIAL DEGREE", N);

  int Nq = N+1; //number of points in 1D
  int Np = Nq*Nq*Nq; //number of points in full cube

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
  ogs.Setup(Nelements*Np, ids, comm, verbose, gpu_aware);

  //make a host gs handle (calls gslib)
  void *gsHandle = gsSetup(comm, Nelements*Np, ids, 0, 0);
  free(ids);

  /*************************
   * Test correctness
   *************************/

  //make an array
  dfloat *q = (dfloat *) malloc(Nelements*Np*sizeof(dfloat));

  //populate an array with the result we expect
  dfloat *qtest = (dfloat *) malloc(Nelements*Np*sizeof(dfloat));

  //fill with ones
  for (dlong n=0;n<Nelements*Np;n++) q[n]=1.0;

  for (dlong n=0;n<Nelements*Np;n++) qtest[n] = q[n];

  //make a device array o_q, copying q from host on creation
  occa::memory o_q = platform.malloc(Nelements*Np*sizeof(dfloat), q);

  //call a gatherScatter operation
  ogs.GatherScatter(o_q);

  //copy back to host
  o_q.copyTo(q);

  gsGatherScatter(qtest, gsHandle);

  dfloat err=0.0;
  for (dlong n=0;n<Nelements*Np;n++) err += fabs(q[n]-qtest[n]);

  dfloat errG=0.0;
  MPI_Reduce(&err, &errG, 1, MPI_DFLOAT, MPI_SUM, 0, comm);

  if (rank==0)
    std::cout << "Error = " << errG << std::endl;

  // close down MPI
  MPI_Finalize();
  return LIBP_SUCCESS;
}

