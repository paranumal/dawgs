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

#ifndef LIBP_MEMORY_HPP
#define LIBP_MEMORY_HPP

#include <cstddef>
#include <memory>
#include <algorithm>
#include <iostream>
#include <typeinfo>
#include <occa.hpp>
#include "utils.hpp"

namespace libp {

template<typename T>
class memory {
  template <typename U> friend class memory;

 private:
  using size_t = std::size_t;
  using ptrdiff_t = std::ptrdiff_t;

  std::shared_ptr<T[]> shrdPtr;
  size_t lngth;
  size_t offset;

 public:
  memory() :
    lngth{0},
    offset{0} {}

  memory(const size_t lngth_) :
    shrdPtr(new T[lngth_]),
    lngth{lngth_},
    offset{0} {}

  memory(const size_t lngth_,
         const T val) :
    shrdPtr(new T[lngth_]),
    lngth{lngth_},
    offset{0} {
    #pragma omp parallel for
    for (size_t i=0;i<lngth;++i) {
      shrdPtr[i] = val;
    }
  }

  /*Conversion constructor*/
  template<typename U>
  memory(const memory<U> &m):
    shrdPtr{std::reinterpret_pointer_cast<T[]>(m.shrdPtr)},
    lngth{m.lngth*sizeof(T)/sizeof(U)},
    offset{m.offset*sizeof(T)/sizeof(U)} {
    // Check that this conversion made sense
    if(lngth*sizeof(U) != m.lngth*sizeof(T)) {
      std::stringstream ss;
      ss << "libp::memory type conversion failed. Trying to convert "
         << m.lngth << " " << sizeof(T) << "-byte words to "
         << lngth << " " << sizeof(U) << "-byte words.";
      LIBP_ABORT(ss.str());
    }
    if(offset*sizeof(U) != m.offset*sizeof(T)) {
      std::stringstream ss;
      ss << "libp::memory type conversion failed. Source memory has offset at "
         << m.lngth << " " << sizeof(T) << "-byte words, destination memory would have offset at"
         << lngth << " " << sizeof(U) << "-byte words.";
      LIBP_ABORT(ss.str());
    }
  }

  memory(const memory<T> &m)=default;
  memory& operator = (const memory<T> &m)=default;
  ~memory()=default;

  void malloc(const size_t lngth_) {
    *this = memory<T>(lngth_);
  }

  void calloc(const size_t lngth_) {
    *this = memory<T>(lngth_, T{0});
  }

  void realloc(const size_t lngth_) {
    memory<T> m(lngth_);
    const ptrdiff_t cnt = std::min(lngth, lngth_);
    m.copyFrom(*this, cnt);
    *this = m;
  }

  memory& swap(memory<T> &m) {
    std::swap(shrdPtr, m.shrdPtr);
    std::swap(lngth, m.lngth);
    std::swap(offset, m.offset);
    return *this;
  }

  T* ptr() {
    return shrdPtr.get()+offset;
  }
  const T* ptr() const {
    return shrdPtr.get()+offset;
  }

  T* begin() {return ptr();}
  T* end() {return ptr() + length();}

  size_t length() const {
    return lngth;
  }

  size_t size() const {
    return lngth*sizeof(T);
  }

  size_t use_count() const {
    return shrdPtr.use_count();
  }

  T& operator[](const ptrdiff_t idx) const {
    return shrdPtr[idx+offset];
  }

  bool operator == (const memory<T> &other) const {
    return (shrdPtr==other.shrdPtr && offset==other.offset);
  }
  bool operator != (const memory<T> &other) const {
    return (shrdPtr!=other.shrdPtr || offset!=other.offset);
  }

  memory<T> operator + (const ptrdiff_t offset_) const {
    return slice(offset_);
  }
  memory<T>& operator += (const ptrdiff_t offset_) {
    *this = slice(offset_);
    return *this;
  }

  memory<T> slice(const ptrdiff_t offset_,
                  const ptrdiff_t count = -1) const {
    memory<T> m(*this);
    m.offset = offset + offset_;
    m.lngth = (count==-1)
                ? (lngth - offset_)
                : count;
    return m;
  }

  /*Copy from raw ptr*/
  void copyFrom(const T* src,
                const ptrdiff_t count = -1,
                const ptrdiff_t offset_ = 0) {

    const ptrdiff_t cnt = (count==-1) ? lngth : count;
    std::copy(src,
              src+cnt,
              ptr()+offset_);
  }

  /*Copy from memory*/
  void copyFrom(const memory<T> src,
                const ptrdiff_t count = -1,
                const ptrdiff_t offset_ = 0) {
    const ptrdiff_t cnt = (count==-1) ? lngth : count;
    std::copy(src.ptr(),
              src.ptr()+cnt,
              ptr()+offset_);
  }

  /*Copy from occa::memory*/
  void copyFrom(const occa::memory &src,
                const ptrdiff_t count = -1,
                const ptrdiff_t destOffset = 0,
                const ptrdiff_t srcOffset = 0,
                const occa::json &props = occa::json()) {
    const ptrdiff_t cnt = (count==-1) ? lngth : count;
    src.copyTo(ptr()+destOffset,
               cnt*sizeof(T),
               srcOffset,
               props);
  }

  void copyFrom(const occa::memory &src,
                const occa::json &props) {
    src.copyTo(ptr(),
               lngth*sizeof(T),
               0,
               props);
  }

  /*Copy to raw pointer*/
  void copyTo(T *dest,
              const ptrdiff_t count = -1,
              const ptrdiff_t offset_ = 0) const {
    const ptrdiff_t cnt = (count==-1) ? lngth : count;
    std::copy(ptr()+offset_,
              ptr()+offset_+cnt,
              dest);
  }

  /*Copy to memory*/
  void copyTo(memory<T> dest,
              const ptrdiff_t count = -1,
              const ptrdiff_t offset_ = 0) const {
    const ptrdiff_t cnt = (count==-1) ? lngth : count;
    std::copy(ptr()+offset_,
              ptr()+offset_+cnt,
              dest.ptr());
  }

  /*Copy to occa::memory*/
  void copyTo(occa::memory &dest,
              const ptrdiff_t count = -1,
              const ptrdiff_t destOffset = 0,
              const ptrdiff_t srcOffset = 0,
              const occa::json &props = occa::json()) const {
    const ptrdiff_t cnt = (count==-1) ? lngth : count;
    dest.copyFrom(ptr()+srcOffset,
               cnt*sizeof(T),
               destOffset,
               props);
  }

  void copyTo(occa::memory &dest,
              const occa::json &props) const {
    dest.copyFrom(ptr(),
               lngth*sizeof(T),
               0,
               props);
  }


  memory<T> clone() const {
    memory<T> m(lngth);
    m.copyFrom(*this);
    return m;
  }


  void free() {
    shrdPtr = nullptr;
    lngth=0;
    offset=0;
  }
};

template <typename T>
std::ostream& operator << (std::ostream &out,
                         const memory<T> &memory) {
  out << "memory - "
      << "type: " << typeid(T).name() << ", "
      << "ptr : " << memory.ptr() << ", "
      << "length : " << memory.length() << ", "
      << "use_count : " << memory.use_count();
  return out;
}

/*Extern declare common instantiations for faster compilation*/
extern template class memory<int>;
extern template class memory<long long int>;
extern template class memory<float>;
extern template class memory<double>;

} //namespace libp

#endif
