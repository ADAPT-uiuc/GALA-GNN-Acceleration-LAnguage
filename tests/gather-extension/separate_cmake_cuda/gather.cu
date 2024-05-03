#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>

#include <torch/script.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <parallel/algorithm>

//#include <ATen/ParallelOpenMP.h>
#include <stdlib.h>           // EXIT_FAILURE
#include <omp.h>
#include <bits/stdc++.h>

// TODO -- Unnecessary -- Torch should handle this.
//int *moveInt(int *data, int size) {
//    int *intOffset;
//    CHECK_CUDA(cudaMalloc((void **) &intOffset,
//                          (size) * sizeof(int)));
//    CHECK_CUDA(cudaMemcpy(intOffset, data,
//                          (size) * sizeof(int),
//                          cudaMemcpyHostToDevice))
//    return intOffset;
//}

// TODO -- Create buffer

// TODO -- Create handle

// TODO -- Create

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



std::vector <at::Tensor> gather_forward(
        torch::Tensor input_dense,
        torch::Tensor offset_graph,
        torch::Tensor columns_graph,
        torch::Tensor value_graph,
        torch::Tensor weights,
        torch::Tensor bias) {
    auto nrows = offset_graph.numel() - 1;
    auto nvals = columns_graph.numel();
    auto full_iden = input_dense.numel();
    auto dcols = full_iden / nrows;

    // // Dense
    // Input
    float *iden_ptr = input_dense.data_ptr<float>();
    // Output
    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);
    auto output_dense = torch::zeros({nrows, dcols}, options);
    float *oden_array = output_dense.data_ptr<float>();

    // Sparse
    int *offset_ptr = offset_graph.data_ptr<int>();
    int *col_ptr = columns_graph.data_ptr<int>();
    float *val_ptr = value_graph.data_ptr<float>();

    float alpha = 1.0f;
    float beta = 1.0f;

    // Create the sparse / dense objects
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CUSPARSE_CHECK(cusparseCreate(&handle))
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, nrows, nrows, nvals,
                      offset_ptr, col_ptr, val_ptr,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, // Need to change these
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // Create dense matrix B
    CUSPARSE_CHECK(cusparseCreateDnMat(&matB, nrows, dcols, dcols, iden_ptr,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW)) // changed
    // Create dense matrix C
    CUSPARSE_CHECK(cusparseCreateDnMat(&matC, nrows, dcols, dcols, oden_array,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW)) // changed

    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_CSR_ALG2,
            &bufferSize));
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    CUSPARSE_CHECK(cusparseSpMM(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                 CUSPARSE_SPMM_CSR_ALG2,
                 dBuffer));

    return {output_dense};
}

TORCH_LIBRARY(gala_ops, m) {
m.def("gather_forward", gather_forward);
}