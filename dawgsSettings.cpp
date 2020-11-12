/*

The MIT License (MIT)

Copyright (c) 2020 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

//settings for dawgs test
dawgsSettings_t::dawgsSettings_t(const int argc, char** argv, MPI_Comm &_comm):
  settings_t(_comm) {

  newSetting("-m", "--mode",
             "THREAD MODEL",
             "CUDA",
             "OCCA's Parallel execution platform",
             {"Serial", "OpenMP", "CUDA", "HIP", "OpenCL"});

  newSetting("-pl", "--platform",
             "PLATFORM NUMBER",
             "0",
             "Parallel platform number (used in OpenCL mode)");

  newSetting("-d", "--device",
             "DEVICE NUMBER",
             "0",
             "Parallel device number");

  newSetting("-nx", "--dimx",
             "LOCAL BOX NX",
             "10",
             "Number of elements in X-dimension");
  newSetting("-ny", "--dimy",
             "LOCAL BOX NY",
             "10",
             "Number of elements in Y-dimension");
  newSetting("-nz", "--dimz",
             "LOCAL BOX NZ",
             "10",
             "Number of elements in Z-dimension");

  newSetting("-NX", "--DIMX",
             "BOX NX",
             "-1",
             "Global number of elements in X-dimension");
  newSetting("-NY", "--DIMY",
             "BOX NY",
             "-1",
             "Global number of elements in Y-dimension");
  newSetting("-NZ", "--DIMZ",
             "BOX NZ",
             "-1",
             "Global number of elements in Z-dimension");

  newSetting("-p", "--degree",
             "POLYNOMIAL DEGREE",
             "4",
             "Degree of polynomial finite element space",
             {"1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"});

  newSetting("-v", "--verbose",
             "VERBOSE",
             "FALSE",
             "Enable verbose output",
             {"TRUE", "FALSE"});

  newSetting("-ga", "--gpu-aware",
             "GPU AWARE",
             "FALSE",
             "Enable GPU aware",
             {"TRUE", "FALSE"});

  newSetting("-cc", "--correctness-check",
             "CORRECTNESS CHECK",
             "FALSE",
             "Check correctness of results using gslib",
             {"TRUE", "FALSE"});

  parseSettings(argc, argv);
}

void dawgsSettings_t::report() {

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank==0) {
    std::cout << "Settings:\n\n";
    reportSetting("THREAD MODEL");

    if (compareSetting("THREAD MODEL","OpenCL"))
      reportSetting("PLATFORM NUMBER");

    if ((size==1)
      &&(compareSetting("THREAD MODEL","CUDA")
          ||compareSetting("THREAD MODEL","HIP")
          ||compareSetting("THREAD MODEL","OpenCL") ))
      reportSetting("DEVICE NUMBER");

    //report the box settings
    reportSetting("BOX NX");
    reportSetting("BOX NY");
    reportSetting("BOX NZ");

    reportSetting("POLYNOMIAL DEGREE");

    reportSetting("GPU AWARE");

    reportSetting("CORRECTNESS CHECK");
  }
}