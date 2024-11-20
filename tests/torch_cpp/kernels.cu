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

// #include "../../src/ops/reordering.h"
// #include "../../src/third_party/rabbit_reorder/rabbit_reordering.h"

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

void printMemoryUsage() {
  size_t freeMem, totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);

  // std::cout << "Free Memory: " << freeMem / (1024 * 1024) << " MB" <<
  // std::endl; std::cout << "Total Memory: " << totalMem / (1024 * 1024) << "
  // MB" << std::endl;
  std::cout << "Used Memory: " << (totalMem - freeMem) / (1024 * 1024) << " MB"
            << std::endl;
}

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

//         for (iT i = work_per_core * c; i < std::min(work_per_core * c,
//         lbs->nrows()); i++) {
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

// Undirected sampled
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel64_undir_sample(float *__restrict__ C,
                                           int *__restrict__ J_indptr_data,
                                           float *__restrict__ B,
                                           int *__restrict__ J_indices_data,
                                           int nrows, int dcols, int nsamples,
                                           int ra, int rb) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    int jmax =
        (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
         J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
    if (jmax > 0) {
      for (int ji = 0; ji < nsamples; ++ji) {
        int j = (ra * ji + rb) % jmax;
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
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel32_undir_sample(float *__restrict__ C,
                                           int *__restrict__ J_indptr_data,
                                           float *__restrict__ B,
                                           int *__restrict__ J_indices_data,
                                           int nrows, int dcols, int nsamples,
                                           int ra, int rb) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    int jmax =
        (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
         J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
    if (jmax > 0) {
      for (int ji = 0; ji < nsamples; ++ji) {
        int j = (ra * ji + rb) % jmax;
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
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel32_offset_undir_sample(
        float *__restrict__ C, int *__restrict__ J_indptr_data,
        float *__restrict__ B, int *__restrict__ J_indices_data, int nrows,
        int dcols, int offset, int nsamples, int ra, int rb) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    int jmax =
        (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
         J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
    for (int ji = 0; ji < nsamples; ++ji) {
      int j = (ra * ji + rb) % jmax;
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
    default_function_kernel_rem_undir_sample(float *__restrict__ C,
                                             int *__restrict__ J_indptr_data,
                                             float *__restrict__ B,
                                             int *__restrict__ J_indices_data,
                                             int nrows, int dcols, int offset,
                                             int nsamples, int ra, int rb) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    int jmax =
        (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
         J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
    if (jmax > 0) {
      for (int ji = 0; ji < nsamples; ++ji) {
        int j = (ra * ji + rb) % jmax;
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
}
// Directed
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel64_sample(float *__restrict__ C,
                                     int *__restrict__ J_indptr_data,
                                     float *__restrict__ A,
                                     float *__restrict__ B,
                                     int *__restrict__ J_indices_data,
                                     int nrows, int dcols, int nsamples, int ra,
                                     int rb) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    int jmax =
        (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
         J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
    for (int ji = 0; ji < nsamples; ++ji) {
      int j = (ra * ji + rb) % jmax;
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
    default_function_kernel32_sample(float *__restrict__ C,
                                     int *__restrict__ J_indptr_data,
                                     float *__restrict__ A,
                                     float *__restrict__ B,
                                     int *__restrict__ J_indices_data,
                                     int nrows, int dcols, int nsamples, int ra,
                                     int rb) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    int jmax =
        (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
         J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
    if (jmax > 0) {
      for (int ji = 0; ji < nsamples; ++ji) {
        int j = (ra * ji + rb) % jmax;
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
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel32_offset_sample(float *__restrict__ C,
                                            int *__restrict__ J_indptr_data,
                                            float *__restrict__ A,
                                            float *__restrict__ B,
                                            int *__restrict__ J_indices_data,
                                            int nrows, int dcols, int offset,
                                            int nsamples, int ra, int rb) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    int jmax =
        (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
         J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
    for (int ji = 0; ji < nsamples; ++ji) {
      int j = (ra * ji + rb) % jmax;
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
    default_function_kernel_rem_sample(float *__restrict__ C,
                                       int *__restrict__ J_indptr_data,
                                       float *__restrict__ A,
                                       float *__restrict__ B,
                                       int *__restrict__ J_indices_data,
                                       int nrows, int dcols, int offset,
                                       int nsamples, int ra, int rb) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    int jmax =
        (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
         J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
    if (jmax > 0) {
      for (int ji = 0; ji < nsamples; ++ji) {
        int j = (ra * ji + rb) % jmax;
        C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
            (((int)blockIdx.y) * 64)) +
           ((int)threadIdx.x)) +
          offset] =
            (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                 (((int)blockIdx.y) * 64)) +
                ((int)threadIdx.x)) +
               offset] +
             (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                    ((int)threadIdx.y))])] *
              B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                      ((int)threadIdx.y))])] *
                   dcols) +
                  (((int)blockIdx.y) * 64)) +
                 ((int)threadIdx.x)) +
                offset]));
      }
    }
  }
}

// Undirected
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel1408_undir(float *__restrict__ C,
                                      int *__restrict__ J_indptr_data,
                                      float *__restrict__ B,
                                      int *__restrict__ J_indices_data,
                                      int nrows, int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
#pragma unroll
      for (int k = 0; k < 44; k++) {
        C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
             (((int)blockIdx.y) * 1408)) +
            ((int)threadIdx.x)) +
           44 * k)] =
            (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                  (((int)blockIdx.y) * 1408)) +
                 ((int)threadIdx.x)) +
                44 * k)] +
             (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                       ((int)threadIdx.y))])] *
                    dcols) +
                   (((int)blockIdx.y) * 1408)) +
                  ((int)threadIdx.x)) +
                 44 * k)]));
      }
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel1024_undir(float *__restrict__ C,
                                      int *__restrict__ J_indptr_data,
                                      float *__restrict__ B,
                                      int *__restrict__ J_indices_data,
                                      int nrows, int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
#pragma unroll
      for (int k = 0; k < 32; k++) {
        C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
             (((int)blockIdx.y) * 1024)) +
            ((int)threadIdx.x)) +
           32 * k)] =
            (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                  (((int)blockIdx.y) * 1024)) +
                 ((int)threadIdx.x)) +
                32 * k)] +
             (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                       ((int)threadIdx.y))])] *
                    dcols) +
                   (((int)blockIdx.y) * 1024)) +
                  ((int)threadIdx.x)) +
                 32 * k)]));
      }
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel576_undir(float *__restrict__ C,
                                     int *__restrict__ J_indptr_data,
                                     float *__restrict__ B,
                                     int *__restrict__ J_indices_data,
                                     int nrows, int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
#pragma unroll
      for (int k = 0; k < 18; k++) {
        C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
             (((int)blockIdx.y) * 576)) +
            ((int)threadIdx.x)) +
           32 * k)] =
            (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                  (((int)blockIdx.y) * 576)) +
                 ((int)threadIdx.x)) +
                32 * k)] +
             (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                       ((int)threadIdx.y))])] *
                    dcols) +
                   (((int)blockIdx.y) * 576)) +
                  ((int)threadIdx.x)) +
                 32 * k)]));
      }
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel512_undir(float *__restrict__ C,
                                     int *__restrict__ J_indptr_data,
                                     float *__restrict__ B,
                                     int *__restrict__ J_indices_data,
                                     int nrows, int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
#pragma unroll
      for (int k = 0; k < 16; k++) {
        C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
             (((int)blockIdx.y) * 512)) +
            ((int)threadIdx.x)) +
           32 * k)] =
            (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                  (((int)blockIdx.y) * 512)) +
                 ((int)threadIdx.x)) +
                32 * k)] +
             (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                       ((int)threadIdx.y))])] *
                    dcols) +
                   (((int)blockIdx.y) * 512)) +
                  ((int)threadIdx.x)) +
                 32 * k)]));
      }
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel256_undir(float *__restrict__ C,
                                     int *__restrict__ J_indptr_data,
                                     float *__restrict__ B,
                                     int *__restrict__ J_indices_data,
                                     int nrows, int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
          (((int)blockIdx.y) * 256)) +
         ((int)threadIdx.x))] =
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
               (((int)blockIdx.y) * 256)) +
              ((int)threadIdx.x))] +
           (B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])] *
                 dcols) +
                (((int)blockIdx.y) * 256)) +
               ((int)threadIdx.x))]));
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
           (((int)blockIdx.y) * 256)) +
          ((int)threadIdx.x)) +
         32)] =
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                (((int)blockIdx.y) * 256)) +
               ((int)threadIdx.x)) +
              32)] +
           (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                     ((int)threadIdx.y))])] *
                  dcols) +
                 (((int)blockIdx.y) * 256)) +
                ((int)threadIdx.x)) +
               32)]));
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
           (((int)blockIdx.y) * 256)) +
          ((int)threadIdx.x)) +
         64)] =
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                (((int)blockIdx.y) * 256)) +
               ((int)threadIdx.x)) +
              64)] +
           (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                     ((int)threadIdx.y))])] *
                  dcols) +
                 (((int)blockIdx.y) * 256)) +
                ((int)threadIdx.x)) +
               64)]));
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
           (((int)blockIdx.y) * 256)) +
          ((int)threadIdx.x)) +
         96)] =
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                (((int)blockIdx.y) * 256)) +
               ((int)threadIdx.x)) +
              96)] +
           (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                     ((int)threadIdx.y))])] *
                  dcols) +
                 (((int)blockIdx.y) * 256)) +
                ((int)threadIdx.x)) +
               96)]));
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
           (((int)blockIdx.y) * 256)) +
          ((int)threadIdx.x)) +
         128)] =
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                (((int)blockIdx.y) * 256)) +
               ((int)threadIdx.x)) +
              128)] +
           (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                     ((int)threadIdx.y))])] *
                  dcols) +
                 (((int)blockIdx.y) * 256)) +
                ((int)threadIdx.x)) +
               128)]));
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
           (((int)blockIdx.y) * 256)) +
          ((int)threadIdx.x)) +
         160)] =
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                (((int)blockIdx.y) * 256)) +
               ((int)threadIdx.x)) +
              160)] +
           (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                     ((int)threadIdx.y))])] *
                  dcols) +
                 (((int)blockIdx.y) * 256)) +
                ((int)threadIdx.x)) +
               160)]));
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
           (((int)blockIdx.y) * 256)) +
          ((int)threadIdx.x)) +
         192)] =
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                (((int)blockIdx.y) * 256)) +
               ((int)threadIdx.x)) +
              192)] +
           (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                     ((int)threadIdx.y))])] *
                  dcols) +
                 (((int)blockIdx.y) * 256)) +
                ((int)threadIdx.x)) +
               192)]));
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
           (((int)blockIdx.y) * 256)) +
          ((int)threadIdx.x)) +
         224)] =
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                (((int)blockIdx.y) * 256)) +
               ((int)threadIdx.x)) +
              224)] +
           (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                     ((int)threadIdx.y))])] *
                  dcols) +
                 (((int)blockIdx.y) * 256)) +
                ((int)threadIdx.x)) +
               224)]));
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel128_undir(float *__restrict__ C,
                                     int *__restrict__ J_indptr_data,
                                     float *__restrict__ B,
                                     int *__restrict__ J_indices_data,
                                     int nrows, int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
          (((int)blockIdx.y) * 128)) +
         ((int)threadIdx.x))] =
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
               (((int)blockIdx.y) * 128)) +
              ((int)threadIdx.x))] +
           (B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])] *
                 dcols) +
                (((int)blockIdx.y) * 128)) +
               ((int)threadIdx.x))]));
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
           (((int)blockIdx.y) * 128)) +
          ((int)threadIdx.x)) +
         32)] =
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                (((int)blockIdx.y) * 128)) +
               ((int)threadIdx.x)) +
              32)] +
           (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                     ((int)threadIdx.y))])] *
                  dcols) +
                 (((int)blockIdx.y) * 128)) +
                ((int)threadIdx.x)) +
               32)]));
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
           (((int)blockIdx.y) * 128)) +
          ((int)threadIdx.x)) +
         64)] =
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                (((int)blockIdx.y) * 128)) +
               ((int)threadIdx.x)) +
              64)] +
           (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                     ((int)threadIdx.y))])] *
                  dcols) +
                 (((int)blockIdx.y) * 128)) +
                ((int)threadIdx.x)) +
               64)]));
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
           (((int)blockIdx.y) * 128)) +
          ((int)threadIdx.x)) +
         96)] =
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                (((int)blockIdx.y) * 128)) +
               ((int)threadIdx.x)) +
              96)] +
           (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                     ((int)threadIdx.y))])] *
                  dcols) +
                 (((int)blockIdx.y) * 128)) +
                ((int)threadIdx.x)) +
               96)]));
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel64_undir(float *__restrict__ C,
                                    int *__restrict__ J_indptr_data,
                                    float *__restrict__ B,
                                    int *__restrict__ J_indices_data, int nrows,
                                    int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    float local_1 = C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                        (((int)blockIdx.y) * 64)) +
                       ((int)threadIdx.x))];
    float local_2 =
        C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
             (((int)blockIdx.y) * 64)) +
            ((int)threadIdx.x)) +
           32)];
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      local_1 =
          local_1 +
          (B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                   ((int)threadIdx.y))])] *
                dcols) +
               (((int)blockIdx.y) * 64)) +
              ((int)threadIdx.x))]);
      local_2 =
          local_2 +
          (B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])] *
                 dcols) +
                (((int)blockIdx.y) * 64)) +
               ((int)threadIdx.x)) +
              32)]);
    }
    C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
        (((int)blockIdx.y) * 64)) +
       ((int)threadIdx.x))] = local_1;
    C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
         (((int)blockIdx.y) * 64)) +
        ((int)threadIdx.x)) +
       32)] = local_2;
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
    default_function_kernel1024(float *__restrict__ C,
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
#pragma unroll
      for (int k = 0; k < 32; k++) {
        C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
             (((int)blockIdx.y) * 1024)) +
            ((int)threadIdx.x)) +
           32 * k)] =
            (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                  (((int)blockIdx.y) * 1024)) +
                 ((int)threadIdx.x)) +
                32 * k)] +
             (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                    ((int)threadIdx.y))])] *
              B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                       ((int)threadIdx.y))])] *
                    dcols) +
                   (((int)blockIdx.y) * 1024)) +
                  ((int)threadIdx.x)) +
                 32 * k)]));
      }
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel512(float *__restrict__ C,
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
#pragma unroll
      for (int k = 0; k < 16; k++) {
        C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
             (((int)blockIdx.y) * 512)) +
            ((int)threadIdx.x)) +
           32 * k)] =
            (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                  (((int)blockIdx.y) * 512)) +
                 ((int)threadIdx.x)) +
                32 * k)] +
             (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                    ((int)threadIdx.y))])] *
              B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                       ((int)threadIdx.y))])] *
                    dcols) +
                   (((int)blockIdx.y) * 512)) +
                  ((int)threadIdx.x)) +
                 32 * k)]));
      }
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel256(float *__restrict__ C,
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
#pragma unroll
      for (int k = 0; k < 8; k++) {
        C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
             (((int)blockIdx.y) * 256)) +
            ((int)threadIdx.x)) +
           32 * k)] =
            (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                  (((int)blockIdx.y) * 256)) +
                 ((int)threadIdx.x)) +
                32 * k)] +
             (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                    ((int)threadIdx.y))])] *
              B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                       ((int)threadIdx.y))])] *
                    dcols) +
                   (((int)blockIdx.y) * 256)) +
                  ((int)threadIdx.x)) +
                 32 * k)]));
      }
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel128(float *__restrict__ C,
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
#pragma unroll
      for (int k = 0; k < 4; k++) {
        C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
             (((int)blockIdx.y) * 128)) +
            ((int)threadIdx.x)) +
           32 * k)] =
            (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
                  (((int)blockIdx.y) * 128)) +
                 ((int)threadIdx.x)) +
                32 * k)] +
             (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                    ((int)threadIdx.y))])] *
              B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                       ((int)threadIdx.y))])] *
                    dcols) +
                   (((int)blockIdx.y) * 128)) +
                  ((int)threadIdx.x)) +
                 32 * k)]));
      }
    }
  }
}
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

// SDDVV
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_sddvv_plus(
        float *__restrict__ C,           // output
        int *__restrict__ J_indptr_data, // index pointer
        float *__restrict__ A,           // input A
        float *__restrict__ B,           // Input B
        int *__restrict__ J_indices_data, int nrows) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine
    for (int j = (int)threadIdx.x; // Not fine. This should increase by 32
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         j += 32) {
      C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =
          (C[(j +
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] +
           (A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] +
            B[(J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                  ((int)threadIdx.y))])])]));
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_sddvv_mult(
        float *__restrict__ C,           // output
        int *__restrict__ J_indptr_data, // index pointer
        float *__restrict__ A,           // input A
        float *__restrict__ B,           // Input B
        int *__restrict__ J_indices_data, int nrows) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine
    for (int j = (int)threadIdx.x; // Not fine. This should increase by 32
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         j += 32) {
      C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =
          (C[(j +
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *
           (A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] *
            B[(J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                  ((int)threadIdx.y))])])]));
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_sddvv_mult_undir(
        float *__restrict__ C,           // output
        int *__restrict__ J_indptr_data, // index pointer
        float *__restrict__ A,           // input A
        float *__restrict__ B,           // Input B
        int *__restrict__ J_indices_data, int nrows) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine
    for (int j = (int)threadIdx.x; // Not fine. This should increase by 32
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         j += 32) {
      C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =
          ((A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] *
            B[(J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                  ((int)threadIdx.y))])])]));
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_sddvv_plus_undir(
        float *__restrict__ C,           // output
        int *__restrict__ J_indptr_data, // index pointer
        float *__restrict__ A,           // input A
        float *__restrict__ B,           // Input B
        int *__restrict__ J_indices_data, int nrows) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine
    for (int j = (int)threadIdx.x; // Not fine. This should increase by 32
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         j += 32) {
      C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =
          (A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] +
           B[(J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                 ((int)threadIdx.y))])])]);
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_sddvv_plus_nowarp(
        float *__restrict__ C,           // output
        int *__restrict__ J_indptr_data, // index pointer
        float *__restrict__ A,           // input A
        float *__restrict__ B,           // Input B
        int *__restrict__ J_indices_data, int nrows) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine
    for (int j = 0; // Not fine. This should increase by 32
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         j += 1) {
      C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =
          (C[(j +
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] +
           (A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] +
            B[(J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                  ((int)threadIdx.y))])])]));
    }
  }
}
// SDDMM
// Undirected
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_sddmm_mult_undir(
        float *__restrict__ C,           // output
        int *__restrict__ J_indptr_data, // index pointer
        float *__restrict__ A,           // input A
        float *__restrict__ B,           // Input B
        int *__restrict__ J_indices_data, int nrows, int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine
    for (int j = (int)threadIdx.x; // Not fine. This should increase by 32
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         j += 32) {
      float local_C = 0;
      for (int k = 0; k < dcols; k++) {
        local_C =
            local_C +
            ((A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + k] *
              B[(J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])]) *
                    dcols +
                k]));
      }
      C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =
          local_C;
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_sddmm_mult_undir_shared(
        float *__restrict__ C,           // output
        int *__restrict__ J_indptr_data, // index pointer
        float *__restrict__ A,           // input A
        float *__restrict__ B,           // Input B
        int *__restrict__ J_indices_data, int nrows, int dcols) {
  extern __shared__ float shared_mem[];

  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine

    for (int k = threadIdx.x; k < dcols; k += 32) {
      if (k < dcols) {
        shared_mem[k] =
            A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + k];
      }
    }
    __syncthreads();

    for (int j = (int)threadIdx.x; // Not fine. This should increase by 32
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         j += 32) {
      float local_C = 0;
      for (int k = 0; k < dcols; k++) {
        local_C =
            local_C +
            ((shared_mem[k] *
              B[(J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])]) *
                    dcols +
                k]));
      }
      C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =
          local_C;
    }
  }
}
// SpMM backward SDDMM
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_spmm_backward_sddmm(
        float *__restrict__ C, // Output dense
        int *__restrict__ J_indptr_data,
        float *__restrict__ A, // Input values
        int *__restrict__ J_indices_data, int nrows, int offset) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    float local_C = 0;
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      local_C = (local_C + (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                  ((int)threadIdx.y))])]));
    }
    C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) +
        (((int)blockIdx.y) * 64)) +
       ((int)threadIdx.x)) +
      offset] = C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) +
                    (((int)blockIdx.y) * 64)) +
                   ((int)threadIdx.x)) +
                  offset] +
                local_C;
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_spmm_backward_sddmm_32(
        float *__restrict__ C, // Output dense
        int *__restrict__ J_indptr_data,
        float *__restrict__ A, // Input values
        int *__restrict__ J_indices_data, int nrows) {
  if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < nrows) {
    float local_C = 1e-12;
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
         ++j) {
      local_C = (local_C + (A[(j + J_indptr_data[((((int)blockIdx.x) * 32) +
                                                  ((int)threadIdx.x))])]));
    }
    C[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] =
        C[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + local_C;
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_softmax_sddvv_undir(
        float *__restrict__ C,           // output (values)
        int *__restrict__ J_indptr_data, // index pointer
        float *__restrict__ A,           // input A
        int *__restrict__ J_indices_data, int nrows) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine
    for (int j = (int)threadIdx.x; // Not fine. This should increase by 32
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         j += 32) {
      C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =
          C[(j +
             J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *
          (A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_softmax_sddvv_undir_4(
        float *__restrict__ C,           // output (values)
        int *__restrict__ J_indptr_data, // index pointer
        float *__restrict__ A,           // input A
        int *__restrict__ J_indices_data, int nrows) {
  if (((((int)blockIdx.x) * 4) + ((int)threadIdx.y)) < nrows) { // This is fine
    for (int j = (int)threadIdx.x; // Not fine. This should increase by 32
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 4) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))]);
         j += 32) {
      C[(j + J_indptr_data[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))])] =
          C[(j +
             J_indptr_data[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))])] *
          (A[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))]);
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_softmax_sddvv_undir_1(
        float *__restrict__ C,           // output (values)
        int *__restrict__ J_indptr_data, // index pointer
        float *__restrict__ A,           // input A
        int *__restrict__ J_indices_data, int nrows) {
  if (((((int)blockIdx.x))) < nrows) { // This is fine
    for (int j = (int)threadIdx.x;     // Not fine. This should increase by 32
         j < (J_indptr_data[(((((int)blockIdx.x))) + 1)] -
              J_indptr_data[((((int)blockIdx.x)))]);
         j += 32) {
      C[(j + J_indptr_data[((((int)blockIdx.x)))])] =
          C[(j + J_indptr_data[((((int)blockIdx.x)))])] *
          (A[((((int)blockIdx.x)))]);
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_mult_sddvv_undir(
        float *__restrict__ C,           // output (values)
        int *__restrict__ J_indptr_data, // index pointer
        float *__restrict__ A,           // input A
        int *__restrict__ J_indices_data, int nrows) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine
    for (int j = (int)threadIdx.x; // Not fine. This should increase by 32
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         j += 32) {
      C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =
          C[(j +
             J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *
          (A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
    }
  }
}
