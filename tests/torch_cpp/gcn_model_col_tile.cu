//
// Created by damitha on 5/12/24.
//
// Define a new Module.
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>

#include <torch/script.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <parallel/algorithm>

//#include <ATen/ParallelOpenMP.h>
#include <stdlib.h>
#include <omp.h>
#include <bits/stdc++.h>

typedef int ind1_t;
typedef int ind2_t;
typedef float val_t;

#include "../../src/utils/mtx_io.h"
#include "../../src/matrix/csrc_matrix.h"
#include "../../src/matrix/dense_matrix.h"
#include "../../src/ops/aggregators.h"
#include "../../src/ops/sparse_matrix_ops.h"
#include "../../src/ops/tiling.h"
#include "../common.h"

#include <torch/torch.h>

//Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DM;
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM;

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


std::vector <at::Tensor> gather_forward_gcn(
        torch::Tensor input_dense,
        torch::Tensor offset_graph,
        torch::Tensor columns_graph,
        torch::Tensor value_graph,
        torch::Tensor bounds,
        int nrows, int segments) {
    auto full_iden = input_dense.numel();
    auto dcols = full_iden / nrows;

    // // Dense
    // Input
    float *iden_ptr = input_dense.data_ptr<float>();
    // Output
    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true).device(torch::kCUDA, 0);
    auto output_dense = torch::zeros({nrows, dcols}, options);
    float *oden_array = output_dense.data_ptr<float>();

    // Sparse
    int *offset_ptr = offset_graph.data_ptr<int>();
    int *col_ptr = columns_graph.data_ptr<int>();
    float *val_ptr = value_graph.data_ptr<float>();
    int *bounds_ptr = bounds.data_ptr<int>();

    float alpha = 1.0f;
    float beta = 1.0f;

    // Create the sparse / dense objects
    cusparseHandle_t handle = NULL;
    cusparseDnMatDescr_t matB, matC;

    cudaDeviceSynchronize();

    CUSPARSE_CHECK(cusparseCreate(&handle));
    // Create dense matrix B
    CUSPARSE_CHECK(cusparseCreateDnMat(&matB, nrows, dcols, dcols, iden_ptr,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW)); // changed
    // Create dense matrix C
    CUSPARSE_CHECK(cusparseCreateDnMat(&matC, nrows, dcols, dcols, oden_array,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW)); // changed
    cudaDeviceSynchronize();
    std::cout << "segments: " << segments << ", nrows: " << nrows << std::endl;

    for (int i = 0; i < segments; i++){
        int i1 = i;
        int start_vals = bounds_ptr[i1 * 2];
        int end_vals = bounds_ptr[i1 * 2 + 1];
        int nvals = end_vals - start_vals;

        cudaDeviceSynchronize();
        std::cout << "nvals: " << start_vals << " " << end_vals << " " << nvals << std::endl;

        cusparseSpMatDescr_t matA;
        CUSPARSE_CHECK(cusparseCreateCsr(&matA, nrows, nrows, nvals,
                                         &offset_ptr[i1 * (nrows + 1)], &col_ptr[start_vals], &val_ptr[start_vals],
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, // Need to change these
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        cudaDeviceSynchronize();
        std::cout << "create CSR works." << std::endl;

        void *dBuffer = NULL;
        size_t bufferSize = 0;
        // allocate an external buffer if needed
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                CUSPARSE_SPMM_CSR_ALG2,
                &bufferSize));
        CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

        cudaDeviceSynchronize();
        std::cout << "Buffer size:" << bufferSize << std::endl;

        CUSPARSE_CHECK(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                    CUSPARSE_SPMM_CSR_ALG2,
                                    dBuffer));

        cudaDeviceSynchronize();
        std::cout << "Works till SpMM." << std::endl;

        CUSPARSE_CHECK(cusparseDestroySpMat(matA));

        cudaDeviceSynchronize();
        std::cout << "Cleans1." << std::endl;

        CUDA_CHECK(cudaFree(dBuffer));

        cudaDeviceSynchronize();
        std::cout << "Cleans2." << std::endl;
        //cudaDeviceSynchronize();
        //std::cout << "a4 " << i << std::endl;
    }

//    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    return {output_dense};
}


struct GCN : torch::nn::Module {

    // Implement the Net's algorithm.
    std::vector<torch::Tensor> forward(torch::Tensor input_dense,
                                       torch::Tensor offset_graph,
                                       torch::Tensor columns_graph,
                                       torch::Tensor value_graph,
                                       torch::Tensor bounds,
                                       int nrows, int segments) {
        return gather_forward_gcn(input_dense, offset_graph, columns_graph, value_graph,  bounds, nrows, segments);
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main(int argc, char **argv) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    std::string path = argv[1];
    iT emb_size = stoi(string(argv[2]));

    // Timing configs
    int num_iters = stoi(string(argv[3]));

    // Column tiling
    iT cols_per_tile = stoi(string(argv[4]));

    // Const settings
    int skip_cache_warmup = 5;

    std::string filename;
    SM adj;
    filename = path;
    readSM<SM>(filename, &adj);

    adj.set_all(1);

    // Adj info
    iT nrows = adj.nrows();
    iT ncols = adj.ncols();
    nT nvals = adj.nvals();

    std::vector<SM *> tiled_adj;
    tiled_adj.push_back(&adj);

    torch::Tensor total_offsets;
    torch::Tensor total_cols;
    torch::Tensor total_vals;
    torch::Tensor total_bounds;

    std::vector<iT> tile_offsets = static_ord_col_breakpoints<SM>(&adj, cols_per_tile);

    iT segments = tile_offsets.size() - 1;

    auto options_int = torch::TensorOptions().dtype(torch::kInt).requires_grad(false);
    auto options_float = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);

    // The first and last value of this should also give the offsets for the columns and vals
    total_offsets = torch::zeros({(adj.nrows() + 1) * (segments)}, options_int);
    total_cols = torch::zeros({adj.nvals()}, options_int);
    total_vals = torch::zeros({adj.nvals()}, options_float);

    total_bounds = torch::zeros({2 * (segments)}, options_int);

    ord_col_tiling_torch(tile_offsets, total_offsets, total_cols, total_vals, total_bounds, &adj);

    iT *offset_ptr = total_offsets.data_ptr<iT>();
    iT *col_ptr = total_cols.data_ptr<iT>();
    vT *val_ptr = total_vals.data_ptr<vT>();

    // Init input with random numbers
    DM input_emb;
    input_emb.build(adj.ncols(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
    for (diT i = 0; i < adj.nrows(); i++) {
        for (dnT j = 0; j < emb_size; j++) {
            input_emb.vals_ptr()[i * emb_size + j] = (dvT) (rand() % 100) / 100;
        }
    }
    input_emb.set_all(1);

    DM out_emb;
    out_emb.build(adj.nrows(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);

    DM out_emb2;
    out_emb2.build(adj.nrows(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);

    auto wsum_aggr = wsumAgg<val_t, val_t, ind2_t>;

    // Comparison for checking if SpMM works correctlu
    out_emb.set_all(0);
    out_emb2.set_all(0);

    // gSpMM(&adj, &input_emb, &out_emb2, wsum_aggr);

    std::cout << adj.nrows() << " " << adj.ncols() << " " << adj.nvals() << std::endl;

    // Create a new Net.
    auto net = std::make_shared<GCN>();

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    iT *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB;

    CUDA_CHECK(cudaMalloc((void **) &dA_csrOffsets,
                          ((nrows + 1) * segments) * sizeof(iT)));
    CUDA_CHECK(cudaMalloc((void **) &dA_columns, nvals * sizeof(iT)));
    CUDA_CHECK(cudaMalloc((void **) &dA_values, nvals * sizeof(vT)));
    CUDA_CHECK(cudaMalloc((void **) &dB, (nrows * emb_size) * sizeof(vT)));

    CUDA_CHECK(cudaMemcpy(dA_csrOffsets, offset_ptr,
                          ((nrows + 1) * segments) * sizeof(iT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_columns, col_ptr, nvals * sizeof(iT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_values, val_ptr, nvals * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, input_emb.vals_ptr(), (nrows * emb_size)  * sizeof(float),
                          cudaMemcpyHostToDevice));

    // CUDA_CHECK(cudaMemcpy(dA_csrOffsets, adj.offset_ptr(),
    //                       (nrows + 1) * sizeof(iT),
    //                       cudaMemcpyHostToDevice));

    // for (iT i = 0; i < segments; i++){
    //   std::cout << "segment:" << i << std::endl;
    //   for (iT j = 0; j < nrows; j++){
    //     if (!(offset_ptr[j + (nrows + 1) * i] <= offset_ptr[j + 1 + (nrows + 1) * i])){
    //       std::cout << "Issue with offset: " << i << " " << j << std::endl;
    //       for (nT e = offset_ptr[j + (nrows + 1) * i]; e < offset_ptr[j + 1 + (nrows + 1) * i]; e++){
    //         if (!(col_ptr[e] == 1 && val_ptr[e] == 1)){
    //           std::cout << "Val / Col: " << e << std::endl;
    //         }
    //       }
    //     }
    //     //std::cout << offset_ptr[j + (nrows + 1) * i] << std::endl;
    //   }
    //   // std::cout << offset_ptr[nrows + (nrows + 1) * i] << std::endl;
    // }
    // CUDA_CHECK(cudaMemcpy(dA_columns, adj.ids_ptr(), nvals * sizeof(iT),
    //                       cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(dA_values, adj.vals_ptr(), nvals * sizeof(float),
    //                       cudaMemcpyHostToDevice));


    auto options_cu_int = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, 0);
    torch::Tensor t_offsets = torch::from_blob(dA_csrOffsets, {(nrows + 1) * segments}, options_cu_int);
    torch::Tensor t_cols = torch::from_blob(dA_columns, {nvals}, options_cu_int);

    auto options_cu_float = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, 0);
    torch::Tensor t_vals = torch::from_blob(dA_values, {nvals}, options_cu_float);
    torch::Tensor t_iden = torch::from_blob(dB, {nrows * emb_size}, options_cu_float);

    double start, end;
    val_t randVal;
    std::vector<double> times_arr;
    for (size_t epoch = 1; epoch <= num_iters; ++epoch) {
        // Reset gradients.
        optimizer.zero_grad();
        // Execute the model on the input data.
        cudaDeviceSynchronize();
        start = get_time();
        torch::Tensor prediction = net->forward(t_iden, t_offsets, t_cols, t_vals, total_bounds, nrows, segments)[0];

        cudaDeviceSynchronize();
        end = get_time();

        if (epoch >= skip_cache_warmup) {
            times_arr.push_back(end - start);
        }

    }

    CUDA_CHECK(cudaFree(dA_csrOffsets));
    CUDA_CHECK(cudaFree(dA_values));
    CUDA_CHECK(cudaFree(dA_columns));
    CUDA_CHECK(cudaFree(dB));

    std::cout << calc_mean(times_arr) << "," << calc_std(times_arr) << std::endl;

}