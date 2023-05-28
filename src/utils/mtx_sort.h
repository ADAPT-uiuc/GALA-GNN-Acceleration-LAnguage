#ifndef _MTX_SORT_H
#define _MTX_SORT_H

#include <omp.h>

#include <cassert>
#include <iostream>
#include <parallel/algorithm>
#include <parallel/numeric>

#include "../utils/threading_utils.h"

#ifdef SparseLibDBG
#define MTX_SORT_DBG
#endif

#ifdef MTX_SORT_DBG
#define MTXSORT_DBG_MSG std::cout << "MTXSORT_" << __LINE__ << ": "
#else
#define MTXSORT_DBG_MSG \
  if (0) std::cout << "MTXSORT_" << __LINE__ << ": "
#endif

uint64_t interleave_uint32_zeros(uint32_t input) {
  uint64_t word = input;
  word = (word ^ (word << 16)) & 0x0000ffff0000ffff;
  word = (word ^ (word << 8)) & 0x00ff00ff00ff00ff;
  word = (word ^ (word << 4)) & 0x0f0f0f0f0f0f0f0f;
  word = (word ^ (word << 2)) & 0x3333333333333333;
  word = (word ^ (word << 1)) & 0x5555555555555555;
  return word;
}

#if defined(__INTEL_COMPILER) & defined(SKYLAKE)
uint64_t mortoncode2(uint32_t r, uint32_t c) {
  uint64_t word11 = r;
  uint64_t word11_int = _pdep_u64(word11, 0xAAAAAAAAAAAAAAAA);
  uint64_t word21 = c;
  uint64_t word21_int = _pdep_u64(word21, 0x5555555555555555);
  return (word11_int | word21_int);
}
#endif

uint64_t interleave_uint32(uint32_t r, uint32_t c) {
#if defined(__INTEL_COMPILER) & defined(SKYLAKE)
  return mortoncode2(r, c);
#else
  return interleave_uint32_zeros(c) | (interleave_uint32_zeros(r) << 1);  // row first
#endif
}

template <typename I_, typename N_, typename ID_>
void count_atomic(I_* arr, N_* counts, ID_ nitems, N_ nvals) {
  double s, e;
  s = get_time();
#pragma omp parallel for
  for (ID_ i = 0; i < nitems; i++) counts[i] = 0;
#pragma omp parallel for
  for (N_ e = 0; e < nvals; e++) {
    __sync_fetch_and_add(&counts[arr[e]], 1);
  }
  e = get_time();
  MTXSORT_DBG_MSG << "TimeCount: " << (e - s) << " secs." << std::endl;
}

template <typename I_, typename N_, typename V_, typename ID_>
void count_sort_place(const ID_* ids, const N_* offsets, N_* wspace, const ID_ nids, const N_ nvals, const I_* row_ids, const I_* col_ids, const V_* vals, I_* row_ids_out, I_* col_ids_out,
                      V_* vals_out) {
  double s, e;
  s = get_time();
#pragma omp parallel for schedule(static, 1024)
  for (ID_ i = 0; i < nids; i++) wspace[i] = offsets[i];

#pragma omp parallel for schedule(static, 1024)
  for (N_ e = 0; e < nvals; e++) {
    N_ index = __sync_fetch_and_add(&wspace[ids[e]], (N_)(1));
    col_ids_out[index] = col_ids[e];
    row_ids_out[index] = row_ids[e];
    if (vals != nullptr) vals_out[index] = vals[e];
  }
#ifdef MTX_SORT_DBG
#pragma omp parallel for schedule(static, 1024)
  for (ID_ i = 0; i < nids; i++) assert(wspace[i] == offsets[i + 1]);

#endif
  e = get_time();
  MTXSORT_DBG_MSG << "TimeCountSortPlace: " << (e - s) << " secs." << std::endl;
}

template <typename N_, typename ID_, typename I1_>
void count_sort_place_1arr(const ID_* ids, const N_* offsets, N_* wspace, const ID_ nids, const N_ nvals, const I1_* arr1, I1_* arr1_c) {
  double s, e;
  s = get_time();
#pragma omp parallel for schedule(static, 1024)
  for (ID_ i = 0; i < nids; i++) wspace[i] = offsets[i];

#pragma omp parallel for schedule(static, 1024)
  for (N_ e = 0; e < nvals; e++) {
    N_ index = __sync_fetch_and_add(&wspace[ids[e]], (N_)(1));
    arr1_c[index] = arr1[e];
    // col_ids_out[index] = col_ids[e];
    // row_ids_out[index] = row_ids[e];
    // if(vals != nullptr) vals_out[index] = vals[e];
  }
#ifdef MTX_SORT_DBG
#pragma omp parallel for schedule(static, 1024)
  for (ID_ i = 0; i < nids; i++) assert(wspace[i] == offsets[i + 1]);

#endif
  e = get_time();
  MTXSORT_DBG_MSG << "TimeCountSortPlace1: " << (e - s) << " secs." << std::endl;
}

template <typename N_, typename ID_, typename I1_, typename I2_>
void count_sort_place_2arr(const ID_* ids, const N_* offsets, N_* wspace, const ID_ nids, const N_ nvals, const I1_* arr1, const I2_* arr2, I1_* arr1_c, I2_* arr2_c) {
  double s, e;
  s = get_time();
#pragma omp parallel for schedule(static, 1024)
  for (ID_ i = 0; i < nids; i++) wspace[i] = offsets[i];

#pragma omp parallel for schedule(static, 1024)
  for (N_ e = 0; e < nvals; e++) {
    N_ index = __sync_fetch_and_add(&wspace[ids[e]], (N_)(1));
    arr1_c[index] = arr1[e];
    arr2_c[index] = arr2[e];
    // col_ids_out[index] = col_ids[e];
    // row_ids_out[index] = row_ids[e];
    // if(vals != nullptr) vals_out[index] = vals[e];
  }
#ifdef MTX_SORT_DBG
#pragma omp parallel for schedule(static, 1024)
  for (ID_ i = 0; i < nids; i++) assert(wspace[i] == offsets[i + 1]);

#endif
  e = get_time();
  MTXSORT_DBG_MSG << "TimeCountSortPlace2: " << (e - s) << " secs." << std::endl;
}

template <typename N_, typename ID_, typename I1_, typename I2_, typename I3_>
void count_sort_place_3arr(const ID_* ids, const N_* offsets, N_* wspace, const ID_ nids, const N_ nvals, const I1_* arr1, const I2_* arr2, const I3_* arr3, I1_* arr1_c, I2_* arr2_c, I3_* arr3_c) {
  double s, e;
  s = get_time();
#pragma omp parallel for schedule(static, 1024)
  for (ID_ i = 0; i < nids; i++) wspace[i] = offsets[i];

#pragma omp parallel for schedule(static, 1024)
  for (N_ e = 0; e < nvals; e++) {
    N_ index = __sync_fetch_and_add(&wspace[ids[e]], (N_)(1));
    arr1_c[index] = arr1[e];
    arr2_c[index] = arr2[e];
    arr3_c[index] = arr3[e];
    // col_ids_out[index] = col_ids[e];
    // row_ids_out[index] = row_ids[e];
    // if(vals != nullptr) vals_out[index] = vals[e];
  }
#ifdef MTX_SORT_DBG
#pragma omp parallel for schedule(static, 1024)
  for (ID_ i = 0; i < nids; i++) assert(wspace[i] == offsets[i + 1]);

#endif
  e = get_time();
  MTXSORT_DBG_MSG << "TimeCountSortPlace2: " << (e - s) << " secs." << std::endl;
}

template <typename I_, typename N_>
void partial_sum(const N_* counts, N_* psum, I_ nitems) {
  double s, e;
  s = get_time();
  psum[0] = 0;
  __gnu_parallel::partial_sum(&(counts[0]), &(counts[0]) + nitems, &(psum[1])); // CHANGED FROM
//  std::partial_sum(&(counts[0]), &(counts[0]) + nitems, &(psum[1]));
  e = get_time();
  MTXSORT_DBG_MSG << "TimePsum: " << (e - s) << " secs." << std::endl;
}

template <typename N_, typename I1_, typename I2_, typename I3_, typename I4_>
N_ qsort_part_4arr(N_ start, N_ end, I1_* arr1, I2_* arr2, I3_* arr3, I4_* arr4) {
  // Pick rightmost element as pivot from the array
  I1_ p1 = arr1[end];
  I2_ p2 = arr2[end];
  I3_ p3 = arr3[end];
  I4_ p4 = arr4[end];

  // elements less than pivot goes to the left of pIndex
  // elements more than pivot goes to the right of pIndex
  // equal elements can go either way
  N_ pIndex = start;
  // each time we finds an element less than or equal to pivot, pIndex
  // is incremented and that element would be placed before the pivot.
  for (N_ i = start; i < end; i++) {
    if (arr1[i] <= p1) {
      std::swap(arr1[i], arr1[pIndex]);
      std::swap(arr2[i], arr2[pIndex]);
      std::swap(arr3[i], arr3[pIndex]);
      std::swap(arr4[i], arr4[pIndex]);
      pIndex++;
    }
  }
  // swap pIndex with Pivot
  std::swap(arr1[pIndex], arr1[end]);
  std::swap(arr2[pIndex], arr2[end]);
  std::swap(arr3[pIndex], arr3[end]);
  std::swap(arr4[pIndex], arr4[end]);

  // return pIndex (index of pivot element)
  return pIndex;
}

template <typename N_, typename I1_, typename I2_, typename I3_>
N_ qsort_part_3arr(N_ start, N_ end, I1_* arr1, I2_* arr2, I3_* arr3) {
  // Pick rightmost element as pivot from the array
  I1_ p1 = arr1[end];
  I2_ p2 = arr2[end];
  I3_ p3 = arr3[end];

  // elements less than pivot goes to the left of pIndex
  // elements more than pivot goes to the right of pIndex
  // equal elements can go either way
  N_ pIndex = start;
  // each time we finds an element less than or equal to pivot, pIndex
  // is incremented and that element would be placed before the pivot.
  for (N_ i = start; i < end; i++) {
    if (arr1[i] <= p1) {
      std::swap(arr1[i], arr1[pIndex]);
      std::swap(arr2[i], arr2[pIndex]);
      std::swap(arr3[i], arr3[pIndex]);
      pIndex++;
    }
  }
  // swap pIndex with Pivot
  std::swap(arr1[pIndex], arr1[end]);
  std::swap(arr2[pIndex], arr2[end]);
  std::swap(arr3[pIndex], arr3[end]);

  // return pIndex (index of pivot element)
  return pIndex;
}

template <typename N_, typename I1_, typename I2_>
N_ qsort_part_2arr(N_ start, N_ end, I1_* arr1, I2_* arr2) {
  // Pick rightmost element as pivot from the array
  I1_ p1 = arr1[end];
  I2_ p2 = arr2[end];
  // elements less than pivot goes to the left of pIndex
  // elements more than pivot goes to the right of pIndex
  // equal elements can go either way
  N_ pIndex = start;
  // each time we finds an element less than or equal to pivot, pIndex
  // is incremented and that element would be placed before the pivot.
  for (N_ i = start; i < end; i++) {
    if (arr1[i] <= p1) {
      std::swap(arr1[i], arr1[pIndex]);
      std::swap(arr2[i], arr2[pIndex]);
      pIndex++;
    }
  }
  // swap pIndex with Pivot
  std::swap(arr1[pIndex], arr1[end]);
  std::swap(arr2[pIndex], arr2[end]);

  // return pIndex (index of pivot element)
  return pIndex;
}
// https://www.techiedelight.com/iterative-implementation-of-quicksort/
template <typename N_, typename I1_>
N_ qsort_part_1arr(N_ start, N_ end, I1_* arr1) {
  // Pick rightmost element as pivot from the array
  I1_ p1 = arr1[end];
  // elements less than pivot goes to the left of pIndex
  // elements more than pivot goes to the right of pIndex
  // equal elements can go either way
  N_ pIndex = start;
  // each time we finds an element less than or equal to pivot, pIndex
  // is incremented and that element would be placed before the pivot.
  for (N_ i = start; i < end; i++) {
    if (arr1[i] <= p1) {
      std::swap(arr1[i], arr1[pIndex]);
      pIndex++;
    }
  }
  // swap pIndex with Pivot
  std::swap(arr1[pIndex], arr1[end]);

  // return pIndex (index of pivot element)
  return pIndex;
}

#include <stack>

// Iterative Quicksort routine
template <typename N_, typename I1_>
void qsort_1arr(N_ n, I1_* arr1) {
  // stack of std::pairs for storing subarray start and end index
  std::stack<std::pair<N_, N_>> stk;

  // get starting and ending index of given array (vector)
  N_ start = 0;
  N_ end = n - (N_)(1);

  // push array start and end index to the stack
  stk.push(std::make_pair(start, end));

  // loop till stack is empty
  while (!stk.empty()) {
    // pop top pair from the list and get sub-array starting
    // and ending indices
    start = stk.top().first, end = stk.top().second;
    stk.pop();

    // rearrange the elements across pivot
    N_ pivot = qsort_part_1arr(start, end, arr1);

    // push sub-array indices containing elements that are
    // less than current pivot to stack
    if (pivot - (N_)(1) > start) {
      stk.push(std::make_pair(start, pivot - (N_)(1)));
    }

    // push sub-array indices containing elements that are
    // more than current pivot to stack
    if (pivot + (N_)(1) < end) {
      stk.push(std::make_pair(pivot + (N_)(1), end));
    }
  }
}

template <typename N_, typename I1_, typename I2_>
void qsort_2arr(N_ n, I1_* arr1, I2_* arr2) {
  // stack of std::pairs for storing subarray start and end index
  std::stack<std::pair<N_, N_>> stk;

  // get starting and ending index of given array (vector)
  N_ start = 0;
  N_ end = n - (N_)(1);

  // push array start and end index to the stack
  stk.push(std::make_pair(start, end));

  // loop till stack is empty
  while (!stk.empty()) {
    // pop top pair from the list and get sub-array starting
    // and ending indices
    start = stk.top().first, end = stk.top().second;
    stk.pop();

    // rearrange the elements across pivot
    N_ pivot = qsort_part_2arr(start, end, arr1, arr2);

    // push sub-array indices containing elements that are
    // less than current pivot to stack
    if (pivot - (N_)(1) > start) {
      stk.push(std::make_pair(start, pivot - (N_)(1)));
    }

    // push sub-array indices containing elements that are
    // more than current pivot to stack
    if (pivot + (N_)(1) < end) {
      stk.push(std::make_pair(pivot + (N_)(1), end));
    }
  }
}

template <typename N_, typename I1_, typename I2_, typename I3_>
void qsort_3arr(N_ n, I1_* arr1, I2_* arr2, I3_* arr3) {
  // stack of std::pairs for storing subarray start and end index
  std::stack<std::pair<N_, N_>> stk;

  // get starting and ending index of given array (vector)
  N_ start = 0;
  N_ end = n - (N_)(1);

  // push array start and end index to the stack
  stk.push(std::make_pair(start, end));

  // loop till stack is empty
  while (!stk.empty()) {
    // pop top pair from the list and get sub-array starting
    // and ending indices
    start = stk.top().first, end = stk.top().second;
    stk.pop();

    // rearrange the elements across pivot
    N_ pivot = qsort_part_3arr(start, end, arr1, arr2, arr3);

    // push sub-array indices containing elements that are
    // less than current pivot to stack
    if (pivot - (N_)(1) > start) {
      stk.push(std::make_pair(start, pivot - (N_)(1)));
    }

    // push sub-array indices containing elements that are
    // more than current pivot to stack
    if (pivot + (N_)(1) < end) {
      stk.push(std::make_pair(pivot + (N_)(1), end));
    }
  }
}

template <typename N_, typename I1_, typename I2_, typename I3_, typename I4_>
void qsort_4arr(N_ n, I1_* arr1, I2_* arr2, I3_* arr3, I4_* arr4) {
  // stack of std::pairs for storing subarray start and end index
  std::stack<std::pair<N_, N_>> stk;

  // get starting and ending index of given array (vector)
  N_ start = 0;
  N_ end = n - (N_)(1);

  // push array start and end index to the stack
  stk.push(std::make_pair(start, end));

  // loop till stack is empty
  while (!stk.empty()) {
    // pop top pair from the list and get sub-array starting
    // and ending indices
    start = stk.top().first, end = stk.top().second;
    stk.pop();

    // rearrange the elements across pivot
    N_ pivot = qsort_part_4arr(start, end, arr1, arr2, arr3, arr4);

    // push sub-array indices containing elements that are
    // less than current pivot to stack
    if (pivot - (N_)(1) > start) {
      stk.push(std::make_pair(start, pivot - (N_)(1)));
    }

    // push sub-array indices containing elements that are
    // more than current pivot to stack
    if (pivot + (N_)(1) < end) {
      stk.push(std::make_pair(pivot + (N_)(1), end));
    }
  }
}

template <typename N_, typename I1_, typename I2_, typename I3_, typename I4_>
void local_qsort_4arr(const N_* offsets, const N_ nitems, const N_ nvals, I1_* arr1, I2_* arr2, I3_* arr3, I4_* arr4) {
  // generic sort function on ids
  double s, e;
  s = get_time();

#pragma omp parallel
  {
#pragma omp for schedule(dynamic, 1)
    for (N_ r = 0; r < nitems; r++) {
      if ((offsets[r + 1] - offsets[r]) > 1) {
        I1_* arr1ptr = arr1 + offsets[r];
        I2_* arr2ptr = arr2 + offsets[r];
        I3_* arr3ptr = arr3 + offsets[r];
        I4_* arr4ptr = arr4 + offsets[r];
        qsort_4arr((offsets[r + 1] - offsets[r]), arr1ptr, arr2ptr, arr3ptr, arr4ptr);
      }
    }
  }

  e = get_time();
  MTXSORT_DBG_MSG << "TimeLocalSort4: " << (e - s) << " secs." << std::endl;
}

template <typename N_, typename I1_, typename I2_, typename I3_>
void local_qsort_3arr(const N_* offsets, const N_ nitems, const N_ nvals, I1_* arr1, I2_* arr2, I3_* arr3) {
  // generic sort function on ids
  double s, e;
  s = get_time();

#pragma omp parallel
  {
#pragma omp for schedule(dynamic, 1)
    for (N_ r = 0; r < nitems; r++) {
      if ((offsets[r + 1] - offsets[r]) > 1) {
        I1_* arr1ptr = arr1 + offsets[r];
        I2_* arr2ptr = arr2 + offsets[r];
        I3_* arr3ptr = arr3 + offsets[r];
        qsort_3arr((offsets[r + 1] - offsets[r]), arr1ptr, arr2ptr, arr3ptr);
      }
    }
  }

  e = get_time();
  MTXSORT_DBG_MSG << "TimeLocalSort3: " << (e - s) << " secs." << std::endl;
}

template <typename N_, typename I1_, typename I2_>
void local_qsort_2arr(const N_* offsets, const N_ nitems, const N_ nvals, I1_* arr1, I2_* arr2) {
  // generic sort function on ids
  double s, e;
  s = get_time();

#pragma omp parallel
  {
#pragma omp for schedule(dynamic, 1)
    for (N_ r = 0; r < nitems; r++) {
      if ((offsets[r + 1] - offsets[r]) > 1) {
        I1_* arr1ptr = arr1 + offsets[r];
        I2_* arr2ptr = arr2 + offsets[r];
        qsort_2arr((offsets[r + 1] - offsets[r]), arr1ptr, arr2ptr);
      }
    }
  }

  e = get_time();
  MTXSORT_DBG_MSG << "TimeLocalSort2: " << (e - s) << " secs." << std::endl;
}

template <typename N_, typename I1_>
void local_qsort_1arr(const N_* offsets, const N_ nitems, const N_ nvals, I1_* arr1) {
  // generic sort function on ids
  double s, e;
  s = get_time();

#pragma omp parallel
  {
#pragma omp for schedule(dynamic, 1)
    for (N_ r = 0; r < nitems; r++) {
      if ((offsets[r + 1] - offsets[r]) > 1) {
        I1_* arr1ptr = arr1 + offsets[r];
        qsort_1arr((offsets[r + 1] - offsets[r]), arr1ptr);
      }
    }
  }

  e = get_time();
  MTXSORT_DBG_MSG << "TimeLocalSort1: " << (e - s) << " secs." << std::endl;
}

template <typename N_, typename I1_, typename I2_, typename I3_, typename I4_>
void dssort_4arr(N_ n, I1_* arr1, I2_* arr2, I3_* arr3, I4_* arr4) {
  N_ start = 0;
  N_ end = n - 1;

  while (start < end) {
    N_ minid = start;
    N_ maxid = end;
    for (N_ s = start; s < end; s++) {
      if (arr1[s] < arr1[minid]) {
        minid = s;
      }
      if (arr1[s] > arr1[maxid]) {
        maxid = s;
      }
    }
    std::swap(arr1[start], arr1[minid]);
    std::swap(arr2[start], arr2[minid]);
    std::swap(arr3[start], arr3[minid]);
    std::swap(arr4[start], arr4[minid]);

    std::swap(arr1[end], arr1[maxid]);
    std::swap(arr2[end], arr2[maxid]);
    std::swap(arr3[end], arr3[maxid]);
    std::swap(arr4[end], arr4[maxid]);

    start++;
    end--;
  }
}
// double selection
template <typename N_, typename I1_, typename I2_, typename I3_, typename I4_>
void local_dssort_4arr(const N_* offsets, const N_ nitems, const N_ nvals, I1_* arr1, I2_* arr2, I3_* arr3, I4_* arr4) {
  // generic sort function on ids
  MTXSORT_DBG_MSG << "Running dssort" << std::endl;
  double s, e;
  s = get_time();

#pragma omp parallel
  {
#pragma omp for schedule(dynamic, 1)
    for (N_ r = 0; r < nitems; r++) {
      if ((offsets[r + 1] - offsets[r]) > 1) {
        I1_* arr1ptr = arr1 + offsets[r];
        I2_* arr2ptr = arr2 + offsets[r];
        I3_* arr3ptr = arr3 + offsets[r];
        I4_* arr4ptr = arr4 + offsets[r];
        dssort_4arr((offsets[r + 1] - offsets[r]), arr1ptr, arr2ptr, arr3ptr, arr4ptr);
      }
    }
  }

  e = get_time();
  MTXSORT_DBG_MSG << "TimeLocalSort4: " << (e - s) << " secs." << std::endl;
}

template <typename N_, typename I1_, typename I2_, typename I3_, typename I4_>
void indexsort_4arr(N_ n, I1_* arr1, I2_* arr2, I3_* arr3, I4_* arr4) {
  N_ start = 0;
  N_ end = n - 1;

  while (start < end) {
    N_ minid = start;
    N_ maxid = end;
    for (N_ s = start; s < end; s++) {
      if (arr1[s] < arr1[minid]) {
        minid = s;
      }
      if (arr1[s] > arr1[maxid]) {
        maxid = s;
      }
    }
    std::swap(arr1[start], arr1[minid]);
    std::swap(arr2[start], arr2[minid]);
    std::swap(arr3[start], arr3[minid]);
    std::swap(arr4[start], arr4[minid]);

    std::swap(arr1[end], arr1[maxid]);
    std::swap(arr2[end], arr2[maxid]);
    std::swap(arr3[end], arr3[maxid]);
    std::swap(arr4[end], arr4[maxid]);

    start++;
    end--;
  }
}
// double selection
template <typename N_, typename I1_, typename I2_, typename I3_, typename I4_>
void local_indexsort_4arr(const N_* offsets, const N_ nitems, const N_ nvals, I1_* arr1, I2_* arr2, I3_* arr3, I4_* arr4) {
  // generic sort function on ids
  MTXSORT_DBG_MSG << "Running dssort" << std::endl;
  double s, e;
  s = get_time();

#pragma omp parallel
  {
#pragma omp for schedule(dynamic, 1)
    for (N_ r = 0; r < nitems; r++) {
      if ((offsets[r + 1] - offsets[r]) > 1) {
        I1_* arr1ptr = arr1 + offsets[r];
        I2_* arr2ptr = arr2 + offsets[r];
        I3_* arr3ptr = arr3 + offsets[r];
        I4_* arr4ptr = arr4 + offsets[r];
        dssort_4arr((offsets[r + 1] - offsets[r]), arr1ptr, arr2ptr, arr3ptr, arr4ptr);
      }
    }
  }

  e = get_time();
  MTXSORT_DBG_MSG << "TimeLocalSort4: " << (e - s) << " secs." << std::endl;
}

// template <typename N_, typename I1_, typename I2_, typename I3_, typename I4_>
// void local_indexsort_4arr(const N_* offsets, const N_ nitems, const N_ nvals, I1_* arr1, I2_* arr2, I3_* arr3, I4_* arr4) {}

template <typename ID_, typename N_, typename K_, typename I1_, typename I2_, typename I3_>
void sort_range3arr(ID_* ids, N_* range, K_ nranges, N_ nvals, I1_* arr1, I2_* arr2, I3_* arr3, I1_* arr1_c, I2_* arr2_c, I3_* arr3_c) {
   // generic sort function on ids
  auto sorter = [&ids](N_ a, N_ b) { return ids[a] < ids[b]; };
  double s, e;
  s = get_time();
  N_ max_nnz = 0;
#pragma omp parallel for reduction(max : max_nnz)
  for (K_ r = 0; r < nranges; r++) {
    N_ nnz = (range[r + 1] - range[r]);
    max_nnz = (max_nnz < nnz) ? nnz : max_nnz;
  }

  #pragma omp parallel
  {
    N_ * indices = new N_[max_nnz];
#pragma omp for schedule(dynamic, 4)
    for (K_ r = 0; r < nranges; r++) {
#pragma omp simd
      for (N_ o = range[r]; o < range[r + 1]; o++) {
        indices[o - range[r]] = o;
      }
      if (range[r + 1] > range[r]) {
        std::sort(indices, indices + (range[r + 1] - range[r]), sorter);
        assert(range[r + 1] >= range[r]);
        arr1_c[range[r]] = arr1[indices[0]];
        arr2_c[range[r]] = arr2[indices[0]];
        arr3_c[range[r]] = arr3[indices[0]];

        for (N_ o = 1; o < range[r + 1] - range[r]; o++) {
          N_ ind = indices[o];
          arr1_c[range[r] + o] = arr1[ind];
          arr2_c[range[r] + o] = arr2[ind];
          arr3_c[range[r] + o] = arr3[ind];
        }
      }
    }
    delete[] indices;
  }
  e = get_time();
  MTXSORT_DBG_MSG << "TimeRangeSort3: " << (e - s) << " secs." << std::endl;
}

template <typename ID_, typename N_, typename K_, typename I1_, typename I2_>
void sort_range2arr(ID_* ids, N_* range, K_ nranges, N_ nvals, I1_* arr1, I2_* arr2, I1_* arr1_c, I2_* arr2_c) {
   // generic sort function on ids
  auto sorter = [&ids](N_ a, N_ b) { return ids[a] < ids[b]; };
  double s, e;
  s = get_time();
  N_ max_nnz = 0;
#pragma omp parallel for reduction(max : max_nnz)
  for (K_ r = 0; r < nranges; r++) {
    N_ nnz = (range[r + 1] - range[r]);
    max_nnz = (max_nnz < nnz) ? nnz : max_nnz;
  }

  #pragma omp parallel
  {
    N_ * indices = new N_[max_nnz];
#pragma omp for schedule(dynamic, 4)
    for (K_ r = 0; r < nranges; r++) {
#pragma omp simd
      for (N_ o = range[r]; o < range[r + 1]; o++) {
        indices[o - range[r]] = o;
      }
      if (range[r + 1] > range[r]) {
        std::sort(indices, indices + (range[r + 1] - range[r]), sorter);
        assert(range[r + 1] >= range[r]);
        arr1_c[range[r]] = arr1[indices[0]];
        arr2_c[range[r]] = arr2[indices[0]];

        for (N_ o = 1; o < range[r + 1] - range[r]; o++) {
          N_ ind = indices[o];
          arr1_c[range[r] + o] = arr1[ind];
          arr2_c[range[r] + o] = arr2[ind];
        }
      }
    }
    delete[] indices;
  }
  e = get_time();
  MTXSORT_DBG_MSG << "TimeRangeSort2: " << (e - s) << " secs." << std::endl;
}

template <typename ID_, typename N_, typename K_, typename I1_>
void sort_range1arr(ID_* ids, N_* range, K_ nranges, N_ nvals, I1_* arr1, I1_* arr1_c) {
   // generic sort function on ids
  auto sorter = [&ids](N_ a, N_ b) { return ids[a] < ids[b]; };
  double s, e;
  s = get_time();
  N_ max_nnz = 0;
#pragma omp parallel for reduction(max : max_nnz)
  for (K_ r = 0; r < nranges; r++) {
    N_ nnz = (range[r + 1] - range[r]);
    max_nnz = (max_nnz < nnz) ? nnz : max_nnz;
  }

  #pragma omp parallel
  {
    N_ * indices = new N_[max_nnz];
#pragma omp for schedule(dynamic, 4)
    for (K_ r = 0; r < nranges; r++) {
#pragma omp simd
      for (N_ o = range[r]; o < range[r + 1]; o++) {
        indices[o - range[r]] = o;
      }
      if (range[r + 1] > range[r]) {
        std::sort(indices, indices + (range[r + 1] - range[r]), sorter);
        assert(range[r + 1] >= range[r]);
        arr1_c[range[r]] = arr1[indices[0]];

        for (N_ o = 1; o < range[r + 1] - range[r]; o++) {
          N_ ind = indices[o];
          arr1_c[range[r] + o] = arr1[ind];
        }
      }
    }
    delete[] indices;
  }
  e = get_time();
  MTXSORT_DBG_MSG << "TimeRangeSort1: " << (e - s) << " secs." << std::endl;
}



#endif