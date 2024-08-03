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

//#include <cuda_runtime.h>
//#include <cusparse_v2.h>
//#include <iostream>

// Error handling macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error in " << __FILE__ << " at line "       \
                      << __LINE__ << ": " << cudaGetErrorString(err)       \
                      << std::endl;                                        \
            exit(err);                                                     \
        }                                                                  \
    } while (0)

/// As is, this WILL NOT WORK.
/// - It loads the ENTIRE matrix into shared memory.
/// - It has some y dimension that comes up magically

// Kernel for SpMM using shared memory and warp-level primitives
__global__ void spmm_kernel_optimized(int *d_row_ptr,
                                      int *d_col_ind,
                                      float *d_values,
                                      float *d_dense_B,
                                      float *d_dense_C,
                                      int num_rows,
                                      int num_cols_B,
                                      int num_cols_C) {
    // allocate block size * the number of columns
    extern __shared__ float shared_B[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Load the ENTIRE dense matrix into shared pace XD. NOT going to work.
    for (int j = threadIdx.y; j < num_cols_C; j += blockDim.y) {
        shared_B[threadIdx.x * num_cols_C + j] = d_dense_B[threadIdx.x * num_cols_C + j];
    }
    __syncthreads();

    if (row < num_rows) {
        for (int j = 0; j < num_cols_C; ++j) {
            float sum = 0.0;
            int row_start = d_row_ptr[row];
            int row_end = d_row_ptr[row + 1];
            for (int idx = row_start; idx < row_end; ++idx) {
                int col = d_col_ind[idx];
                float val = d_values[idx];
                sum += val * shared_B[col * num_cols_C + j];
            }
            d_dense_C[row * num_cols_C + j] = sum;
        }
    }
}

void spmm_optimized(int *h_row_ptr,
                    int *h_col_ind,
                    float *h_values,
                    float *h_dense_B,
                    float *h_dense_C,
                    int num_rows,
                    int num_cols_B,
                    int num_cols_C) {
    int nnz = h_row_ptr[num_rows];

    // Allocate device memory
    int *d_row_ptr, *d_col_ind;
    float *d_values, *d_dense_B, *d_dense_C;

    CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_col_ind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dense_B, num_cols_B * num_cols_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dense_C, num_rows * num_cols_C * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, h_col_ind, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dense_B, h_dense_B, num_cols_B * num_cols_C * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel with shared memory
    int blockSize = 32;
    int gridSize = (num_rows + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * num_cols_C * sizeof(float);
    spmm_kernel_optimized<<<gridSize, blockSize, sharedMemSize>>>(d_row_ptr,
                                                                  d_col_ind,
                                                                  d_values,
                                                                  d_dense_B,
                                                                  d_dense_C,
                                                                  num_rows,
                                                                  num_cols_B,
                                                                  num_cols_C);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_dense_C, d_dense_C, num_rows * num_cols_C * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_ind));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_dense_B));
    CUDA_CHECK(cudaFree(d_dense_C));
}

int main() {
    // Example usage
    int h_row_ptr[] = {0, 2, 4};
    int h_col_ind[] = {0, 1, 0, 2};
    float h_values[] = {1.0, 2.0, 3.0, 4.0};
    float h_dense_B[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float h_dense_C[6] = {0.0};

    int num_rows = 2;
    int num_cols_B = 3;
    int num_cols_C = 3;

    spmm_optimized(h_row_ptr, h_col_ind, h_values, h_dense_B, h_dense_C, num_rows, num_cols_B, num_cols_C);

    return 0;
}


//Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DM;
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM;

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

    float alpha = 1.0f;
    float beta = 1.0f;

    // Create the sparse / dense objects
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, nrows, nrows, nvals,
                                     offset_ptr, col_ptr, val_ptr,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, // Need to change these
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // Create dense matrix B
    CUSPARSE_CHECK(cusparseCreateDnMat(&matB, nrows, dcols, dcols, iden_ptr,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW)); // changed
    // Create dense matrix C
    CUSPARSE_CHECK(cusparseCreateDnMat(&matC, nrows, dcols, dcols, oden_array,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW)); // changed

    // allocate an external buffer if needed
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
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

    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    CUDA_CHECK(cudaFree(dBuffer));

    return {output_dense};
}


struct GCN : torch::nn::Module {

    // Implement the Net's algorithm.
    std::vector<torch::Tensor> forward(torch::Tensor input_dense,
                          torch::Tensor offset_graph,
                          torch::Tensor columns_graph,
                          torch::Tensor value_graph) {
        return gather_forward_gcn(input_dense, offset_graph, columns_graph, value_graph);
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