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

void factor3(const int size, int &size_x, int &size_y, int &size_z);

void dawgsReference(dfloat* qtest, platform_t &platform, settings_t& settings){

  //number of MPI ranks
  int size = platform.size;

  //global MPI rank
  int rank = platform.rank;

  // find a factorization size = size_x*size_y*size_z such that
  //  size_x>=size_y>=size_z are all 'close' to one another
  int size_x, size_y, size_z;
  factor3(size, size_x, size_y, size_z);

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

  //get polynomial degree
  int N;
  settings.getSetting("POLYNOMIAL DEGREE", N);

  int Nq = N+1; //number of points in 1D
  int Np = Nq*Nq*Nq; //number of points in full cube

  //fill qtest with expected values after gatherScatter
  for (int K=0;K<nz;K++) {
    for (int J=0;J<ny;J++) {
      for (int I=0;I<nx;I++) {

        dfloat *qtest_e = qtest + (I + J*nx + K*nx*ny)*Np;

        //cube interiors are 1.0
        for (int k=1;k<Nq-1;k++) {
          for (int j=1;j<Nq-1;j++) {
            for (int i=1;i<Nq-1;i++) {
              qtest_e[i+j*Nq+k*Nq*Nq] = 1.0;
            }
          }
        }

        // face nodes are 2.0
        for (int j=1;j<Nq-1;j++) {
          for (int i=1;i<Nq-1;i++) {
            //bottom and top
            qtest_e[i+j*Nq+     0*Nq*Nq] = 2.0;
            qtest_e[i+j*Nq+(Nq-1)*Nq*Nq] = 2.0;
            //front and back
            qtest_e[i+     0*Nq+j*Nq*Nq] = 2.0;
            qtest_e[i+(Nq-1)*Nq+j*Nq*Nq] = 2.0;
            //left and right
            qtest_e[     0+i*Nq+j*Nq*Nq] = 2.0;
            qtest_e[(Nq-1)+i*Nq+j*Nq*Nq] = 2.0;
          }
        }

        //edge nodes are 4.0
        for (int i=1;i<Nq-1;i++) {
          qtest_e[i+     0*Nq+     0*Nq*Nq] = 4.0;
          qtest_e[i+(Nq-1)*Nq+     0*Nq*Nq] = 4.0;
          qtest_e[i+     0*Nq+(Nq-1)*Nq*Nq] = 4.0;
          qtest_e[i+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 4.0;

          qtest_e[     0+i*Nq+     0*Nq*Nq] = 4.0;
          qtest_e[(Nq-1)+i*Nq+     0*Nq*Nq] = 4.0;
          qtest_e[     0+i*Nq+(Nq-1)*Nq*Nq] = 4.0;
          qtest_e[(Nq-1)+i*Nq+(Nq-1)*Nq*Nq] = 4.0;

          qtest_e[     0+     0*Nq+i*Nq*Nq] = 4.0;
          qtest_e[(Nq-1)+     0*Nq+i*Nq*Nq] = 4.0;
          qtest_e[     0+(Nq-1)*Nq+i*Nq*Nq] = 4.0;
          qtest_e[(Nq-1)+(Nq-1)*Nq+i*Nq*Nq] = 4.0;
        }

        //corners are 8.0
        qtest_e[     0+     0*Nq+     0*Nq*Nq] = 8.0;
        qtest_e[(Nq-1)+     0*Nq+     0*Nq*Nq] = 8.0;
        qtest_e[     0+(Nq-1)*Nq+     0*Nq*Nq] = 8.0;
        qtest_e[(Nq-1)+(Nq-1)*Nq+     0*Nq*Nq] = 8.0;
        qtest_e[     0+     0*Nq+(Nq-1)*Nq*Nq] = 8.0;
        qtest_e[(Nq-1)+     0*Nq+(Nq-1)*Nq*Nq] = 8.0;
        qtest_e[     0+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 8.0;
        qtest_e[(Nq-1)+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 8.0;
      }
    }
  }

  //boundaries make this a little complicated so let's go case-by-case
  if (rank_z==0) { //bottom is boundary
    for (int J=0;J<ny;J++) {
      for (int I=0;I<nx;I++) {
        dfloat *qtest_e = qtest + (I + J*nx + 0*nx*ny)*Np;
        // face nodes are 1.0
        for (int j=1;j<Nq-1;j++) {
          for (int i=1;i<Nq-1;i++) {
            qtest_e[i+j*Nq+     0*Nq*Nq] = 1.0;
          }
        }
        //edge nodes are 2.0
        for (int i=1;i<Nq-1;i++) {
          qtest_e[i+     0*Nq+     0*Nq*Nq] = 2.0;
          qtest_e[i+(Nq-1)*Nq+     0*Nq*Nq] = 2.0;
          qtest_e[     0+i*Nq+     0*Nq*Nq] = 2.0;
          qtest_e[(Nq-1)+i*Nq+     0*Nq*Nq] = 2.0;
        }

        //corners are 4.0
        qtest_e[     0+     0*Nq+     0*Nq*Nq] = 4.0;
        qtest_e[(Nq-1)+     0*Nq+     0*Nq*Nq] = 4.0;
        qtest_e[     0+(Nq-1)*Nq+     0*Nq*Nq] = 4.0;
        qtest_e[(Nq-1)+(Nq-1)*Nq+     0*Nq*Nq] = 4.0;
      }
    }
  }
  if (rank_z==size_z-1) { //top is boundary
    for (int J=0;J<ny;J++) {
      for (int I=0;I<nx;I++) {
        dfloat *qtest_e = qtest + (I + J*nx + (nz-1)*nx*ny)*Np;
        // face nodes are 1.0
        for (int j=1;j<Nq-1;j++) {
          for (int i=1;i<Nq-1;i++) {
            qtest_e[i+j*Nq+(Nq-1)*Nq*Nq] = 1.0;
          }
        }
        //edge nodes are 2.0
        for (int i=1;i<Nq-1;i++) {
          qtest_e[     i+     0*Nq+(Nq-1)*Nq*Nq] = 2.0;
          qtest_e[     i+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 2.0;
          qtest_e[     0+     i*Nq+(Nq-1)*Nq*Nq] = 2.0;
          qtest_e[(Nq-1)+     i*Nq+(Nq-1)*Nq*Nq] = 2.0;
        }

        //corners are 4.0
        qtest_e[     0+     0*Nq+(Nq-1)*Nq*Nq] = 4.0;
        qtest_e[(Nq-1)+     0*Nq+(Nq-1)*Nq*Nq] = 4.0;
        qtest_e[     0+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 4.0;
        qtest_e[(Nq-1)+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 4.0;
      }
    }
  }

  if (rank_y==0) { //front is boundary
    for (int K=0;K<nz;K++) {
      for (int I=0;I<nx;I++) {
        dfloat *qtest_e = qtest + (I + 0*nx + K*nx*ny)*Np;
        // face nodes are 1.0
        for (int k=1;k<Nq-1;k++) {
          for (int i=1;i<Nq-1;i++) {
            qtest_e[i+     0*Nq+k*Nq*Nq] = 1.0;
          }
        }
        //edge nodes are 2.0
        for (int i=1;i<Nq-1;i++) {
          qtest_e[     i+     0*Nq+     0*Nq*Nq] = 2.0;
          qtest_e[     i+     0*Nq+(Nq-1)*Nq*Nq] = 2.0;
          qtest_e[     0+     0*Nq+     i*Nq*Nq] = 2.0;
          qtest_e[(Nq-1)+     0*Nq+     i*Nq*Nq] = 2.0;
        }

        //corners are 4.0
        qtest_e[     0+     0*Nq+     0*Nq*Nq] = 4.0;
        qtest_e[(Nq-1)+     0*Nq+     0*Nq*Nq] = 4.0;
        qtest_e[     0+     0*Nq+(Nq-1)*Nq*Nq] = 4.0;
        qtest_e[(Nq-1)+     0*Nq+(Nq-1)*Nq*Nq] = 4.0;
      }
    }
  }
  if (rank_y==size_y-1) { //back is boundary
    for (int K=0;K<nz;K++) {
      for (int I=0;I<nx;I++) {
        dfloat *qtest_e = qtest + (I + (ny-1)*nx + K*nx*ny)*Np;
        // face nodes are 1.0
        for (int k=1;k<Nq-1;k++) {
          for (int i=1;i<Nq-1;i++) {
            qtest_e[i+(Nq-1)*Nq+k*Nq*Nq] = 1.0;
          }
        }
        //edge nodes are 2.0
        for (int i=1;i<Nq-1;i++) {
          qtest_e[     i+(Nq-1)*Nq+     0*Nq*Nq] = 2.0;
          qtest_e[     i+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 2.0;
          qtest_e[     0+(Nq-1)*Nq+     i*Nq*Nq] = 2.0;
          qtest_e[(Nq-1)+(Nq-1)*Nq+     i*Nq*Nq] = 2.0;
        }

        //corners are 4.0
        qtest_e[     0+(Nq-1)*Nq+     0*Nq*Nq] = 4.0;
        qtest_e[(Nq-1)+(Nq-1)*Nq+     0*Nq*Nq] = 4.0;
        qtest_e[     0+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 4.0;
        qtest_e[(Nq-1)+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 4.0;
      }
    }
  }

  if (rank_x==0) { //left is boundary
    for (int K=0;K<nz;K++) {
      for (int J=0;J<ny;J++) {
        dfloat *qtest_e = qtest + (0 + J*nx + K*nx*ny)*Np;
        // face nodes are 1.0
        for (int k=1;k<Nq-1;k++) {
          for (int j=1;j<Nq-1;j++) {
            qtest_e[0+j*Nq+k*Nq*Nq] = 1.0;
          }
        }
        //edge nodes are 2.0
        for (int i=1;i<Nq-1;i++) {
          qtest_e[0+     i*Nq+     0*Nq*Nq] = 2.0;
          qtest_e[0+     i*Nq+(Nq-1)*Nq*Nq] = 2.0;
          qtest_e[0+     0*Nq+     i*Nq*Nq] = 2.0;
          qtest_e[0+(Nq-1)*Nq+     i*Nq*Nq] = 2.0;
        }

        //corners are 4.0
        qtest_e[0+     0*Nq+     0*Nq*Nq] = 4.0;
        qtest_e[0+(Nq-1)*Nq+     0*Nq*Nq] = 4.0;
        qtest_e[0+     0*Nq+(Nq-1)*Nq*Nq] = 4.0;
        qtest_e[0+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 4.0;
      }
    }
  }
  if (rank_x==size_x-1) { //right is boundary
    for (int K=0;K<nz;K++) {
      for (int J=0;J<ny;J++) {
        dfloat *qtest_e = qtest + ((nx-1) + J*nx + K*nx*ny)*Np;
        // face nodes are 1.0
        for (int k=1;k<Nq-1;k++) {
          for (int j=1;j<Nq-1;j++) {
            qtest_e[(Nq-1)+j*Nq+k*Nq*Nq] = 1.0;
          }
        }
        //edge nodes are 2.0
        for (int i=1;i<Nq-1;i++) {
          qtest_e[(Nq-1)+     i*Nq+     0*Nq*Nq] = 2.0;
          qtest_e[(Nq-1)+     i*Nq+(Nq-1)*Nq*Nq] = 2.0;
          qtest_e[(Nq-1)+     0*Nq+     i*Nq*Nq] = 2.0;
          qtest_e[(Nq-1)+(Nq-1)*Nq+     i*Nq*Nq] = 2.0;
        }

        //corners are 4.0
        qtest_e[(Nq-1)+     0*Nq+     0*Nq*Nq] = 4.0;
        qtest_e[(Nq-1)+(Nq-1)*Nq+     0*Nq*Nq] = 4.0;
        qtest_e[(Nq-1)+     0*Nq+(Nq-1)*Nq*Nq] = 4.0;
        qtest_e[(Nq-1)+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 4.0;
      }
    }
  }

  if (rank_z==0 && rank_y==0) { //bottom-front edge is boundary
    for (int I=0;I<nx;I++) {
      dfloat *qtest_e = qtest + (I + 0*nx + 0*nx*ny)*Np;

      //edge nodes are 1.0
      for (int i=1;i<Nq-1;i++) {
        qtest_e[i+     0*Nq+     0*Nq*Nq] = 1.0;
      }

      //corners are 2.0
      qtest_e[     0+     0*Nq+     0*Nq*Nq] = 2.0;
      qtest_e[(Nq-1)+     0*Nq+     0*Nq*Nq] = 2.0;
    }
  }
  if (rank_z==size_z-1 && rank_y==0) { //top-front edge is boundary
    for (int I=0;I<nx;I++) {
      dfloat *qtest_e = qtest + (I + 0*nx + (nz-1)*nx*ny)*Np;

      //edge nodes are 1.0
      for (int i=1;i<Nq-1;i++) {
        qtest_e[i+     0*Nq+(Nq-1)*Nq*Nq] = 1.0;
      }

      //corners are 2.0
      qtest_e[     0+     0*Nq+(Nq-1)*Nq*Nq] = 2.0;
      qtest_e[(Nq-1)+     0*Nq+(Nq-1)*Nq*Nq] = 2.0;
    }
  }
  if (rank_z==0 && rank_y==size_y-1) { //bottom-back edge is boundary
    for (int I=0;I<nx;I++) {
      dfloat *qtest_e = qtest + (I + (ny-1)*nx + 0*nx*ny)*Np;

      //edge nodes are 1.0
      for (int i=1;i<Nq-1;i++) {
        qtest_e[i+(Nq-1)*Nq+     0*Nq*Nq] = 1.0;
      }

      //corners are 2.0
      qtest_e[     0+(Nq-1)*Nq+     0*Nq*Nq] = 2.0;
      qtest_e[(Nq-1)+(Nq-1)*Nq+     0*Nq*Nq] = 2.0;
    }
  }
  if (rank_z==size_z-1 && rank_y==size_y-1) { //top-back edge is boundary
    for (int I=0;I<nx;I++) {
      dfloat *qtest_e = qtest + (I + (ny-1)*nx + (nz-1)*nx*ny)*Np;

      //edge nodes are 1.0
      for (int i=1;i<Nq-1;i++) {
        qtest_e[i+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 1.0;
      }

      //corners are 2.0
      qtest_e[     0+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 2.0;
      qtest_e[(Nq-1)+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 2.0;
    }
  }

  if (rank_z==0 && rank_x==0) { //bottom-left edge is boundary
    for (int J=0;J<ny;J++) {
      dfloat *qtest_e = qtest + (0 + J*nx + 0*nx*ny)*Np;

      //edge nodes are 1.0
      for (int i=1;i<Nq-1;i++) {
        qtest_e[0+i*Nq+     0*Nq*Nq] = 1.0;
      }

      //corners are 2.0
      qtest_e[0+     0*Nq+     0*Nq*Nq] = 2.0;
      qtest_e[0+(Nq-1)*Nq+     0*Nq*Nq] = 2.0;
    }
  }
  if (rank_z==size_z-1 && rank_x==0) { //top-left edge is boundary
    for (int J=0;J<ny;J++) {
      dfloat *qtest_e = qtest + (0 + J*nx + (nz-1)*nx*ny)*Np;

      //edge nodes are 1.0
      for (int i=1;i<Nq-1;i++) {
        qtest_e[0+i*Nq+(Nq-1)*Nq*Nq] = 1.0;
      }

      //corners are 2.0
      qtest_e[0+     0*Nq+(Nq-1)*Nq*Nq] = 2.0;
      qtest_e[0+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 2.0;
    }
  }
  if (rank_z==0 && rank_x==size_x-1) { //bottom-right edge is boundary
    for (int J=0;J<ny;J++) {
      dfloat *qtest_e = qtest + ((nx-1) + J*nx + 0*nx*ny)*Np;

      //edge nodes are 1.0
      for (int i=1;i<Nq-1;i++) {
        qtest_e[(Nq-1)+i*Nq+     0*Nq*Nq] = 1.0;
      }

      //corners are 2.0
      qtest_e[(Nq-1)+     0*Nq+     0*Nq*Nq] = 2.0;
      qtest_e[(Nq-1)+(Nq-1)*Nq+     0*Nq*Nq] = 2.0;
    }
  }
  if (rank_z==size_z-1 && rank_x==size_x-1) { //top-right edge is boundary
    for (int J=0;J<ny;J++) {
      dfloat *qtest_e = qtest + ((nx-1) + J*nx + (nz-1)*nx*ny)*Np;

      //edge nodes are 1.0
      for (int i=1;i<Nq-1;i++) {
        qtest_e[(Nq-1)+i*Nq+(Nq-1)*Nq*Nq] = 1.0;
      }

      //corners are 2.0
      qtest_e[(Nq-1)+     0*Nq+(Nq-1)*Nq*Nq] = 2.0;
      qtest_e[(Nq-1)+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 2.0;
    }
  }

  if (rank_y==0 && rank_x==0) { //front-left edge is boundary
    for (int K=0;K<nz;K++) {
      dfloat *qtest_e = qtest + (0 + 0*nx + K*nx*ny)*Np;

      //edge nodes are 1.0
      for (int i=1;i<Nq-1;i++) {
        qtest_e[0+0*Nq+i*Nq*Nq] = 1.0;
      }

      //corners are 2.0
      qtest_e[0+0*Nq+     0*Nq*Nq] = 2.0;
      qtest_e[0+0*Nq+(Nq-1)*Nq*Nq] = 2.0;
    }
  }
  if (rank_y==size_y-1 && rank_x==0) { //back-left edge is boundary
    for (int K=0;K<nz;K++) {
      dfloat *qtest_e = qtest + (0 + (ny-1)*nx + K*nx*ny)*Np;

      //edge nodes are 1.0
      for (int i=1;i<Nq-1;i++) {
        qtest_e[0+(Nq-1)*Nq+i*Nq*Nq] = 1.0;
      }

      //corners are 2.0
      qtest_e[0+(Nq-1)*Nq+     0*Nq*Nq] = 2.0;
      qtest_e[0+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 2.0;
    }
  }
  if (rank_y==0 && rank_x==size_x-1) { //front-right edge is boundary
    for (int K=0;K<nz;K++) {
      dfloat *qtest_e = qtest + ((nx-1) + 0*nx + K*nx*ny)*Np;

      //edge nodes are 1.0
      for (int i=1;i<Nq-1;i++) {
        qtest_e[(Nq-1)+0*Nq+i*Nq*Nq] = 1.0;
      }

      //corners are 2.0
      qtest_e[(Nq-1)+0*Nq+     0*Nq*Nq] = 2.0;
      qtest_e[(Nq-1)+0*Nq+(Nq-1)*Nq*Nq] = 2.0;
    }
  }
  if (rank_y==size_y-1 && rank_x==size_x-1) { //back-right edge is boundary
    for (int K=0;K<nz;K++) {
      dfloat *qtest_e = qtest + ((nx-1) + (ny-1)*nx + K*nx*ny)*Np;

      //edge nodes are 1.0
      for (int i=1;i<Nq-1;i++) {
        qtest_e[(Nq-1)+(Nq-1)*Nq+i*Nq*Nq] = 1.0;
      }

      //corners are 2.0
      qtest_e[(Nq-1)+(Nq-1)*Nq+     0*Nq*Nq] = 2.0;
      qtest_e[(Nq-1)+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 2.0;
    }
  }

  //corners
  if (rank_z==0 && rank_y==0 && rank_x==0) {
    dfloat *qtest_e = qtest + (0 + 0*nx + 0*nx*ny)*Np;
    qtest_e[0+0*Nq+0*Nq*Nq] = 1.0;
  }
  if (rank_z==size_z-1 && rank_y==0 && rank_x==0) {
    dfloat *qtest_e = qtest + (0 + 0*nx + (nz-1)*nx*ny)*Np;
    qtest_e[0+0*Nq+(Nq-1)*Nq*Nq] = 1.0;
  }
  if (rank_z==0 && rank_y==size_y-1 && rank_x==0) {
    dfloat *qtest_e = qtest + (0 + (ny-1)*nx + 0*nx*ny)*Np;
    qtest_e[0+(Nq-1)*Nq+0*Nq*Nq] = 1.0;
  }
  if (rank_z==size_z-1 && rank_y==size_y-1 && rank_x==0) {
    dfloat *qtest_e = qtest + (0 + (ny-1)*nx + (nz-1)*nx*ny)*Np;
    qtest_e[0+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 1.0;
  }
  if (rank_z==0 && rank_y==0 && rank_x==size_x-1) {
    dfloat *qtest_e = qtest + ((nx-1) + 0*nx + 0*nx*ny)*Np;
    qtest_e[(Nq-1)+0*Nq+0*Nq*Nq] = 1.0;
  }
  if (rank_z==size_z-1 && rank_y==0 && rank_x==size_x-1) {
    dfloat *qtest_e = qtest + ((nx-1) + 0*nx + (nz-1)*nx*ny)*Np;
    qtest_e[(Nq-1)+0*Nq+(Nq-1)*Nq*Nq] = 1.0;
  }
  if (rank_z==0 && rank_y==size_y-1 && rank_x==size_x-1) {
    dfloat *qtest_e = qtest + ((nx-1) + (ny-1)*nx + 0*nx*ny)*Np;
    qtest_e[(Nq-1)+(Nq-1)*Nq+0*Nq*Nq] = 1.0;
  }
  if (rank_z==size_z-1 && rank_y==size_y-1 && rank_x==size_x-1) {
    dfloat *qtest_e = qtest + ((nx-1) + (ny-1)*nx + (nz-1)*nx*ny)*Np;
    qtest_e[(Nq-1)+(Nq-1)*Nq+(Nq-1)*Nq*Nq] = 1.0;
  }
}

