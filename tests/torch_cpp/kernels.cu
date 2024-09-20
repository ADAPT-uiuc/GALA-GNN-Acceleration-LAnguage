#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>
#include <torch/script.h>

#include <cmath>
#include <iostream>
#include <parallel/algorithm>
#include <vector>

// #include <ATen/ParallelOpenMP.h>
#include <bits/stdc++.h>
#include <omp.h>
#include <stdlib.h>

#include <torch/torch.h>

#include "../../src/matrix/csrc_matrix.h"
#include "../../src/matrix/dense_matrix.h"
#include "../../src/ops/aggregators.h"
#include "../../src/ops/sparse_matrix_ops.h"
#include "../../src/ops/tiling.h"
#include "../../src/utils/mtx_io.h"
#include "../common.h"

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUSPARSE_CHECK(func)                                                   \
  do {                                                                         \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


// template<class DM>
// int max_arr(const DM *lbs) {
//     typedef typename DM::itype iT; // Node IDs
//     typedef typename DM::ntype nT; // Edge IDs
//     typedef typename DM::vtype vT; // Value of node

//     int cores = 64;
//     auto mins = (vT *) alloca(cores * sizeof(vT));
//     auto maxs = (vT *) alloca(cores * sizeof(vT));

//     iT work_per_core = (iT)(lbs->nrows() - 1)/cores + 1;

// #pragma omp parallel for schedule(static)
//     for (int c = 0; c < cores; c++) {
//         maxs[c] = 0;
//         // mins[c] = lbs->nrows();

//         for (iT i = work_per_core * c; i < std::min(work_per_core * c, lbs->nrows()); i++) {
//             vT val = lbs->vals_ptr()[i];
//             // if (val < mins[c]) {
//             //     mins[c] = val;
//             // }
//             if (val > maxs[c]) {
//                 maxs[c] = val;
//             }
//         }
//     }

//     int max_val = 0;
//     for (int c = 0; c < cores; c++) {
//         if (max_val < maxs[c]) {
//             max_val = (int)maxs[c];
//         }
//     }
//     return max_val;
// }



// Undirected
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel64_undir(float *__restrict__ C,
                                    int *__restrict__ J_indptr_data,
                                    float *__restrict__ B,
                                    int *__restrict__ J_indices_data, int nrows,
                                    int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
          (((int)blockIdx.y) * 64)) +
         ((int)threadIdx.x))] =
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
               (((int)blockIdx.y) * 64)) +
              ((int)threadIdx.x))] +
           (B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])] *
                 dcols) +
                (((int)blockIdx.y) * 64)) +
               ((int)threadIdx.x))]));
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
           (((int)blockIdx.y) * 64)) +
          ((int)threadIdx.x)) +
         32)] =
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                (((int)blockIdx.y) * 64)) +
               ((int)threadIdx.x)) +
              32)] +
           (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                     ((int)threadIdx.y))])] *
                  dcols) +
                 (((int)blockIdx.y) * 64)) +
                ((int)threadIdx.x)) +
               32)]));
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel32_undir(float *__restrict__ C,
                                    int *__restrict__ J_indptr_data,
                                    float *__restrict__ B,
                                    int *__restrict__ J_indices_data, int nrows,
                                    int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
          (((int)blockIdx.y) * 32)) +
         ((int)threadIdx.x))] =
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
               (((int)blockIdx.y) * 32)) +
              ((int)threadIdx.x))] +
           (B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])] *
                 dcols) +
                (((int)blockIdx.y) * 32)) +
               ((int)threadIdx.x))]));
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel32_offset_undir(float *__restrict__ C,
                                           int *__restrict__ J_indptr_data,
                                           float *__restrict__ B,
                                           int *__restrict__ J_indices_data,
                                           int nrows, int dcols, int offset) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
          (((int)blockIdx.y) * 32)) +
         ((int)threadIdx.x)) +
        offset] =
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
               (((int)blockIdx.y) * 32)) +
              ((int)threadIdx.x)) +
             offset] +
           (B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])] *
                 dcols) +
                (((int)blockIdx.y) * 32)) +
               ((int)threadIdx.x)) +
              offset]));
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_rem_undir(float *__restrict__ C,
                                      int *__restrict__ J_indptr_data,
                                      float *__restrict__ B,
                                      int *__restrict__ J_indices_data,
                                      int nrows, int dcols, int offset) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
          (((int)blockIdx.y) * 32)) +
         ((int)threadIdx.x)) +
        offset] =
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
               (((int)blockIdx.y) * 32)) +
              ((int)threadIdx.x)) +
             offset] +
           (B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])] *
                 dcols) +
                (((int)blockIdx.y) * 32)) +
               ((int)threadIdx.x)) +
              offset]));
    }
  }
}

// Directed
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel64(float *__restrict__ C,
                              int *__restrict__ J_indptr_data,
                              float *__restrict__ A, float *__restrict__ B,
                              int *__restrict__ J_indices_data, int nrows,
                              int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
          (((int)blockIdx.y) * 64)) +
         ((int)threadIdx.x))] =
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
               (((int)blockIdx.y) * 64)) +
              ((int)threadIdx.x))] +
           (A[(j +
               J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *
            B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])] *
                 dcols) +
                (((int)blockIdx.y) * 64)) +
               ((int)threadIdx.x))]));
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
           (((int)blockIdx.y) * 64)) +
          ((int)threadIdx.x)) +
         32)] =
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                (((int)blockIdx.y) * 64)) +
               ((int)threadIdx.x)) +
              32)] +
           (A[(j +
               J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *
            B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                     ((int)threadIdx.y))])] *
                  dcols) +
                 (((int)blockIdx.y) * 64)) +
                ((int)threadIdx.x)) +
               32)]));
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel32(float *__restrict__ C,
                              int *__restrict__ J_indptr_data,
                              float *__restrict__ A, float *__restrict__ B,
                              int *__restrict__ J_indices_data, int nrows,
                              int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
          (((int)blockIdx.y) * 64)) +
         ((int)threadIdx.x))] =
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
               (((int)blockIdx.y) * 64)) +
              ((int)threadIdx.x))] +
           (A[(j +
               J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *
            B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])] *
                 dcols) +
                (((int)blockIdx.y) * 64)) +
               ((int)threadIdx.x))]));
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel32_offset(float *__restrict__ C,
                                     int *__restrict__ J_indptr_data,
                                     float *__restrict__ A,
                                     float *__restrict__ B,
                                     int *__restrict__ J_indices_data,
                                     int nrows, int dcols, int offset) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
          (((int)blockIdx.y) * 64)) +
         ((int)threadIdx.x)) +
        offset] =
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
               (((int)blockIdx.y) * 64)) +
              ((int)threadIdx.x)) +
             offset] +
           (A[(j +
               J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *
            B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])] *
                 dcols) +
                (((int)blockIdx.y) * 64)) +
               ((int)threadIdx.x)) +
              offset]));
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_rem(float *__restrict__ C,
                                int *__restrict__ J_indptr_data,
                                float *__restrict__ A, float *__restrict__ B,
                                int *__restrict__ J_indices_data, int nrows,
                                int dcols, int offset) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
          (((int)blockIdx.y) * 64)) +
         ((int)threadIdx.x)) +
        offset] =
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
               (((int)blockIdx.y) * 64)) +
              ((int)threadIdx.x)) +
             offset] +
           (A[(j +
               J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *
            B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])] *
                 dcols) +
                (((int)blockIdx.y) * 64)) +
               ((int)threadIdx.x)) +
              offset]));
    }
  }
}

// With IF condition
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_if(float *__restrict__ C,
                               int *__restrict__ J_indptr_data,
                               float *__restrict__ A, float *__restrict__ B,
                               int *__restrict__ J_indices_data, int nrows,
                               int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    if ((((((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) < dcols) {
      //      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
      //      (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))] = 0.000000e+00f;
      for (int j = 0;
           j < (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) +
                               1)] -
                J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
           ++j) {
        C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
            (((int)blockIdx.y) * 64)) +
           ((int)threadIdx.x))] =
            (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                 (((int)blockIdx.y) * 64)) +
                ((int)threadIdx.x))] +
             (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                    ((int)threadIdx.y))])] *
              B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                      ((int)threadIdx.y))])] *
                   dcols) +
                  (((int)blockIdx.y) * 64)) +
                 ((int)threadIdx.x))]));
      }
    }

    if ((((((int)blockIdx.y) * 64)) + ((int)threadIdx.x) + 32) < dcols) {
      //      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
      //      (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 32)] =
      //      0.000000e+00f;
      for (int j = 0;
           j < (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) +
                               1)] -
                J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
           ++j) {
        C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
             (((int)blockIdx.y) * 64)) +
            ((int)threadIdx.x)) +
           32)] =
            (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                  (((int)blockIdx.y) * 64)) +
                 ((int)threadIdx.x)) +
                32)] +
             (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                    ((int)threadIdx.y))])] *
              B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                       ((int)threadIdx.y))])] *
                    dcols) +
                   (((int)blockIdx.y) * 64)) +
                  ((int)threadIdx.x)) +
                 32)]));
      }
    }
  }
}