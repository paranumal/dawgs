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

#ifndef PLATFORM_HPP
#define PLATFORM_HPP

#define LIBP_MAJOR_VERSION 0
#define LIBP_MINOR_VERSION 5
#define LIBP_PATCH_VERSION 0
#define LIBP_VERSION       00500
#define LIBP_VERSION_STR   "0.5.0"

#include "core.hpp"
#include "settings.hpp"

namespace libp {

namespace internal {

class iplatform_t {
public:
  settings_t& settings;
  occa::properties props;

  iplatform_t(settings_t& _settings):
    settings(_settings) {
  }
};

} //namespace internal


class platform_t {
public:
  MPI_Comm comm = MPI_COMM_NULL;
  std::shared_ptr<internal::iplatform_t> iplatform;

  occa::device device;

  int rank=0, size=0;

  platform_t()=default;

  platform_t(settings_t& settings) {

    iplatform = std::make_shared<internal::iplatform_t>(settings);

    comm = settings.comm.comm();

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank==0) {
      std::cout << "\n";
      std::cout << "\033[1m";
      std::cout << " _ _ _     ____                                             _ \n";
      std::cout << "| (_) |__ |  _ \\ __ _ _ __ __ _ _ __  _   _ _ __ ___   __ _| |\n";
      std::cout << "| | | '_ \\| |_) / _` | '__/ _` | '_ \\| | | | '_ ` _ \\ / _` | |\n";
      std::cout << "| | | |_) |  __/ (_| | | | (_| | | | | |_| | | | | | | (_| | |\n";
      std::cout << "|_|_|_.__/|_|   \\__,_|_|  \\__,_|_| |_|\\__,_|_| |_| |_|\\__,_|_|\n";
      std::cout << "\033[0m";
      std::cout << "\n";
      std::cout << "Version: " LIBP_VERSION_STR " \n";
      std::cout << "Contributing developers: Noel Chalmers, Ali Karakus, Kasia Swirydowicz,\n";
      std::cout << "                         Anthony Austin, & Tim Warburton\n";
      std::cout << "\n";
    }

    DeviceConfig();
    DeviceProperties();
  }

  platform_t(const platform_t &other)=default;
  platform_t& operator = (const platform_t &other)=default;

  bool isInitialized() {
    return (iplatform!=nullptr);
  }

  void assertInitialized() {
    if(!isInitialized()) {
      LIBP_ABORT("Platform not initialized.");
    }
  }

  occa::kernel buildKernel(std::string fileName, std::string kernelName,
                           occa::properties& kernelInfo);

  template <typename T>
  deviceMemory<T> malloc(const size_t count,
                         const occa::properties &prop = occa::properties()) {
    assertInitialized();
    return deviceMemory<T>(device.malloc<T>(count, prop));
  }

  template <typename T>
  deviceMemory<T> malloc(const size_t count,
                         const libp::memory<T> src,
                         const occa::properties &prop = occa::properties()) {
    assertInitialized();
    return deviceMemory<T>(device.malloc<T>(count, src.ptr(), prop));
  }

  template <typename T>
  deviceMemory<T> malloc(const libp::memory<T> src,
                         const occa::properties &prop = occa::properties()) {
    assertInitialized();
    return deviceMemory<T>(device.malloc<T>(src.length(), src.ptr(), prop));
  }

  template <typename T>
  pinnedMemory<T> hostMalloc(const size_t count){
    assertInitialized();
    occa::properties hostProp;
    hostProp["host"] = true;
    return pinnedMemory<T>(device.malloc<T>(count, nullptr, hostProp));
  }

  template <typename T>
  pinnedMemory<T> hostMalloc(const size_t count,
                             const libp::memory<T> src){
    assertInitialized();
    occa::properties hostProp;
    hostProp["host"] = true;
    return pinnedMemory<T>(device.malloc<T>(count, src.ptr(), hostProp));
  }

  template <typename T>
  pinnedMemory<T> hostMalloc(const libp::memory<T> src){
    assertInitialized();
    occa::properties hostProp;
    hostProp["host"] = true;
    return pinnedMemory<T>(device.malloc<T>(src.length(), src.ptr(), hostProp));
  }

  settings_t& settings() {
    assertInitialized();
    return iplatform->settings;
  }

  occa::properties& props() {
    assertInitialized();
    return iplatform->props;
  }

private:
  void DeviceConfig();
  void DeviceProperties();

};

} //namespace libp

#endif
