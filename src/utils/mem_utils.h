#ifndef _MEM_UTILS_H
#define _MEM_UTILS_H

#include <stddef.h>

#define DEL_1D_ARR(A) \
  if (A != nullptr) { \
    delete[] A;       \
    A = nullptr;      \
  }

template <typename T_>
void alloc_t(T_*& ptr, size_t size, int type = 0) {
  ptr = new T_[size];
}

template <typename T_>
void free_t(T_*& ptr, size_t size, int type = 0) {
  if (ptr != nullptr) {
    delete[] ptr;
    ptr = nullptr;
  }
}
#endif