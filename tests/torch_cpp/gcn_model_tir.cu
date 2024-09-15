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


// extern "C" __global__ void __launch_bounds__(256) default_function_kernel32(float* __restrict__ C,
//                                                                              int* __restrict__ J_indptr_data,
//                                                                              float* __restrict__ A,
//                                                                              float* __restrict__ B,
//                                                                              int* __restrict__ J_indices_data,
//                                                                              int nrows) {
//   if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
//     C[(((((int)blockIdx.x) * 256) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] = 0.000000e+00f;
//     for (int j = 0; j < (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] - J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]); ++j) {
//       C[(((((int)blockIdx.x) * 256) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] = (C[(((((int)blockIdx.x) * 256) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] + (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] * B[((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] * 32) + ((int)threadIdx.x))]));
//     }
//   }
// }

 extern "C" __global__ void __launch_bounds__(256) default_function_kernel64(float* __restrict__ C,
                                                                            int* __restrict__ J_indptr_data,
                                                                            float* __restrict__ A,
                                                                            float* __restrict__ B,
                                                                            int* __restrict__ J_indices_data,
                                                                            int nrows,
                                                                            int dcols) {
   if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
     C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))] = 0.000000e+00f;
     C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 32)] = 0.000000e+00f;
     for (int j = 0; j < (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] - J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]); ++j) {
       C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))] = (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))] + (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] * B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] * dcols) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))]));
       C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 32)] = (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 32)] + (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] * B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] * dcols) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 32)]));
     }
   }
 }

  extern "C" __global__ void __launch_bounds__(256) default_function_kernel_rem(float* __restrict__ C,
                                                                            int* __restrict__ J_indptr_data,
                                                                            float* __restrict__ A,
                                                                            float* __restrict__ B,
                                                                            int* __restrict__ J_indices_data,
                                                                            int nrows,
                                                                            int dcols,
                                                                            int offset) {
   if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
     C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + offset] = 0.000000e+00f;
     for (int j = 0; j < (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] - J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]); ++j) {
       C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + offset] = (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + offset] + (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] * B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] * dcols) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + offset]));
     }
   }
 }

extern "C" __global__ void __launch_bounds__(256) default_function_kernel_if(float* __restrict__ C,
                                                                           int* __restrict__ J_indptr_data,
                                                                           float* __restrict__ A,
                                                                           float* __restrict__ B,
                                                                           int* __restrict__ J_indices_data,
                                                                           int nrows,
                                                                           int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    if ((((((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) < dcols){
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))] = 0.000000e+00f;
      for (int j = 0; j < (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] - J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]); ++j) {
        C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))] = (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))] + (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] * B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] * dcols) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))]));
      }
    }

    if ((((((int)blockIdx.y) * 64)) + ((int)threadIdx.x) + 32) < dcols){
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 32)] = 0.000000e+00f;
      for (int j = 0; j < (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] - J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]); ++j) {
        C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 32)] = (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 32)] + (A[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] * B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] * dcols) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 32)]));
      }
    }
  }
}


std::vector <at::Tensor> gather_forward_gcn(
        torch::Tensor input_dense,
        torch::Tensor offset_graph,
        torch::Tensor columns_graph,
        torch::Tensor value_graph) {
    auto nrows = offset_graph.numel() - 1;
    auto nvals = columns_graph.numel();
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

//	dim3 gridDim((int)nrows / 8, ((int)dcols + 1) / 64);
//    dim3 blockDim(32, 8);
//    default_function_kernel_if<<<gridDim, blockDim>>>(oden_array,
//                                                      offset_ptr,
//                                                      val_ptr,
//                                                      iden_ptr,
//                                                      col_ptr,
//                                                      nrows,
//                                                      dcols);

   dim3 gridDim(((int)nrows + 1) / 8, (int)dcols / 64);
   dim3 blockDim(32, 8);
   default_function_kernel64<<<gridDim, blockDim>>>(oden_array,
                                                    offset_ptr,
                                                    val_ptr,
                                                    iden_ptr,
                                                    col_ptr,
                                                    nrows,
                                                    dcols);

   if ((dcols % 64) > 0) {
      dim3 gridDim_rem((int)nrows / 8, 1);
      dim3 blockDim_rem(dcols % 64, 8);
      default_function_kernel_rem<<<gridDim_rem, blockDim_rem>>>(oden_array,
                                                                 offset_ptr,
                                                                 val_ptr,
                                                                 iden_ptr,
                                                                 col_ptr,
                                                                 nrows,
                                                                 dcols,
                                                                 ((int)dcols / 64) * 64);
   }


    // Have a section that does till after 32, then another for the rest
    // Test out what adding an if condition would do to the overall execution

//    float alpha = 1.0f;
//    float beta = 1.0f;
//
//    // Create the sparse / dense objects
//    cusparseHandle_t handle = NULL;
//    cusparseSpMatDescr_t matA;
//    cusparseDnMatDescr_t matB, matC;
//    void *dBuffer = NULL;
//    size_t bufferSize = 0;
//
//    CUSPARSE_CHECK(cusparseCreate(&handle));
//    CUSPARSE_CHECK(cusparseCreateCsr(&matA, nrows, nrows, nvals,
//                                     offset_ptr, col_ptr, val_ptr,
//                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, // Need to change these
//                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
//    // Create dense matrix B
//    CUSPARSE_CHECK(cusparseCreateDnMat(&matB, nrows, dcols, dcols, iden_ptr,
//                                       CUDA_R_32F, CUSPARSE_ORDER_ROW)); // changed
//    // Create dense matrix C
//    CUSPARSE_CHECK(cusparseCreateDnMat(&matC, nrows, dcols, dcols, oden_array,
//                                       CUDA_R_32F, CUSPARSE_ORDER_ROW)); // changed
//
//    // allocate an external buffer if needed
//    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
//            handle,
//            CUSPARSE_OPERATION_NON_TRANSPOSE,
//            CUSPARSE_OPERATION_NON_TRANSPOSE,
//            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
//            CUSPARSE_SPMM_CSR_ALG2,
//            &bufferSize));
//    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
//
//    CUSPARSE_CHECK(cusparseSpMM(handle,
//                                CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
//                                CUSPARSE_SPMM_CSR_ALG2,
//                                dBuffer));
//
//    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
//    CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
//    CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
//    CUSPARSE_CHECK(cusparseDestroy(handle));
//    CUDA_CHECK(cudaFree(dBuffer));

    return {output_dense};
}


struct GCN : torch::nn::Module {

    // Implement the Net's algorithm.
    std::vector<torch::Tensor> forward(torch::Tensor input_dense,
                          torch::Tensor offset_graph,
                          torch::Tensor columns_graph,
                          torch::Tensor value_graph) {
        return gather_forward_gcn(input_dense, offset_graph, columns_graph, value_graph);

//        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
//        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
//        x = torch::relu(fc2->forward(x));
//        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
//        return x;
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

    //gSpMM(&adj, &input_emb, &out_emb2, wsum_aggr);

    std::cout << adj.nrows() << " " << adj.ncols() << " " << adj.nvals() << std::endl;

    // Create a new Net.
    auto net = std::make_shared<GCN>();

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    int *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB;

    CUDA_CHECK(cudaMalloc((void **) &dA_csrOffsets,
                          (nrows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **) &dA_columns, nvals * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **) &dA_values, nvals * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &dB, (nrows * emb_size) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA_csrOffsets, adj.offset_ptr(),
                          (nrows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_columns, adj.ids_ptr(), nvals * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_values, adj.vals_ptr(), nvals * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, input_emb.vals_ptr(), (nrows * emb_size)  * sizeof(float),
                          cudaMemcpyHostToDevice));

    auto options_cu_int = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, 0);
    torch::Tensor t_offsets = torch::from_blob(dA_csrOffsets, {nrows + 1}, options_cu_int);
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
//        std::vector<torch::Tensor> prediction = net->forward(t_iden, t_offsets, t_cols, t_vals);
        torch::Tensor prediction = net->forward(t_iden, t_offsets, t_cols, t_vals)[0];

        cudaDeviceSynchronize();
        end = get_time();

        if (epoch >= skip_cache_warmup) {
            times_arr.push_back(end - start);
        }

        randVal = prediction[nrows - 1][emb_size - 1].item<val_t>();


//        for (int x = 0; x < nrows; x++){
//            for (int y = 0; y < emb_size; y++){
//                if (prediction[x][y].item<val_t>()!= out_emb2.vals_ptr()[x * emb_size + y]) {
//                    std::cout << "The results don't match at: " << x << "," << y << ":  " << prediction[x][y].item<val_t>() << ", "
//                              << out_emb2.vals_ptr()[x * emb_size + y] << std::endl;
//                    break;
//                }
//            }
//        }

// Compute a loss value to judge the prediction of our model.
//        torch::Tensor loss = torch::nll_loss(prediction, batch.target);
//        // Compute gradients of the loss w.r.t. the parameters of our model.
//        loss.backward();
//        // Update the parameters based on the calculated gradients.
//        optimizer.step();
//        // Output the loss and checkpoint every 100 batches.
//        if (++batch_index % 100 == 0) {
//            std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
//                      << " | Loss: " << loss.item<float>() << std::endl;
//            // Serialize your model periodically as a checkpoint.
//            torch::save(net, "net.pt");
//        }

        // Print the results of the precompute function

    }

    CUDA_CHECK(cudaFree(dA_csrOffsets));
    CUDA_CHECK(cudaFree(dA_values));
    CUDA_CHECK(cudaFree(dA_columns));
    CUDA_CHECK(cudaFree(dB));

    std::cout << calc_mean(times_arr) << "," << calc_std(times_arr) << "|" << randVal << std::endl;
}