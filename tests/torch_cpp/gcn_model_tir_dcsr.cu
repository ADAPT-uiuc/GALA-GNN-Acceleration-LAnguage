//
// Created by damitha on 5/12/24.
//
// Define a new Module.
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

typedef int ind1_t;
typedef int ind2_t;
typedef float val_t;

#include <torch/torch.h>

#include "../../src/matrix/csrc_matrix.h"
#include "../../src/matrix/dense_matrix.h"
#include "../../src/ops/aggregators.h"
#include "../../src/ops/sparse_matrix_ops.h"
#include "../../src/ops/tiling.h"
#include "../../src/utils/mtx_io.h"
#include "../common.h"

// Dense matrix with double values.
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

// Undirected
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel64_undir(float *__restrict__ C,
                                    int *__restrict__ J_indptr_data,
                                    float *__restrict__ B,
                                    int *__restrict__ J_indices_data,
                                    int *__restrict__ J_rows_data, int nrows,
                                    int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((J_rows_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] * dcols +
          (((int)blockIdx.y) * 64)) +
         ((int)threadIdx.x))] =
          (C[((J_rows_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] *
                   dcols +
               (((int)blockIdx.y) * 64)) +
              ((int)threadIdx.x))] +
           (B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                    ((int)threadIdx.y))])] *
                 dcols) +
                (((int)blockIdx.y) * 64)) +
               ((int)threadIdx.x))]));
      C[(((J_rows_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] * dcols +
           (((int)blockIdx.y) * 64)) +
          ((int)threadIdx.x)) +
         32)] =
          (C[(((J_rows_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] *
                    dcols +
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
                                    int *__restrict__ J_indices_data,
                                    int *__restrict__ J_rows_data, int nrows,
                                    int dcols) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((J_rows_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] * dcols +
          (((int)blockIdx.y) * 32)) +
         ((int)threadIdx.x))] =
          (C[((J_rows_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] *
                   dcols +
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
                                           int *__restrict__ J_rows_data,
                                           int nrows, int dcols, int offset) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((J_rows_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] * dcols +
          (((int)blockIdx.y) * 32)) +
         ((int)threadIdx.x)) +
        offset] =
          (C[((J_rows_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] *
                   dcols +
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
                                      int *__restrict__ J_indptr_data, // reset
                                      float *__restrict__ B,
                                      int *__restrict__ J_indices_data,
                                      int *__restrict__ J_rows_data, int nrows,
                                      int dcols, int offset) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    for (int j = 0;
         j <
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
         ++j) {
      C[((J_rows_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] * dcols +
          (((int)blockIdx.y) * 32)) +
         ((int)threadIdx.x)) +
        offset] =
          (C[((J_rows_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] *
                   dcols +
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

std::vector<at::Tensor>
gather_forward_gcn(torch::Tensor input_dense, torch::Tensor offset_graph,
                   torch::Tensor rows_graph, torch::Tensor columns_graph,
                   torch::Tensor value_graph, torch::Tensor ranges, int nrows,
                   int segments, bool is_directed) {
  auto full_iden = input_dense.numel();
  auto dcols = full_iden / nrows;

  // // Dense
  // Input
  float *iden_ptr = input_dense.data_ptr<float>();
  // Output
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat)
                     .requires_grad(true)
                     .device(torch::kCUDA, 0);
  auto output_dense = torch::zeros({nrows, dcols}, options);
  float *oden_array = output_dense.data_ptr<float>();

  // Sparse
  int *offset_ptr = offset_graph.data_ptr<int>();
  int *row_ptr = rows_graph.data_ptr<int>();
  int *col_ptr = columns_graph.data_ptr<int>();
  float *val_ptr = value_graph.data_ptr<float>();
  int *ranges_ptr = ranges.data_ptr<int>();

  for (int i = 0; i < segments; i++) {
    int i1 = i;
    int start_rows = ranges_ptr[i1 * 2];
    int end_rows = ranges_ptr[i1 * 2 + 1];
    int nrows_seg = end_rows - start_rows;

    // std::cout << dcols << " :a: " << start_rows << " " << end_rows << " " << nrows_seg << std::endl;

    // std::cout << "row ids: " << rows_graph[start_rows].item<int>() << " " << rows_graph[end_rows - 1].item<int>() << std::endl;

    // std::cout << new_offset_ptr_vec[row_bounds[nth_tile * 2]] << " " << new_offset_ptr_vec[row_bounds[nth_tile * 2] + 1] << std::endl;
    // std::cout << new_offset_ptr_vec[row_bounds[nth_tile * 2 + 1] - 1] << " " << new_offset_ptr_vec[row_bounds[nth_tile * 2 + 1]] << std::endl;


    cudaStream_t stream1, stream2, stream3;

    if ((int)dcols / 64) {
      cudaStreamCreate(&stream1);
      dim3 gridDim(((int)(nrows_seg - 1) / 8) + 1, (int)dcols / 64);
      dim3 blockDim(32, 8);
      default_function_kernel64_undir<<<gridDim, blockDim, 0, stream1>>>(
          oden_array, &offset_ptr[start_rows], iden_ptr, col_ptr,
          &row_ptr[start_rows], nrows_seg, dcols);

      if ((dcols % 64) > 32) {
        cudaStreamCreate(&stream2);
        dim3 gridDim_rem(((int)(nrows_seg - 1) / 8) + 1, 1);
        dim3 blockDim_rem(32, 8);
        default_function_kernel32_offset_undir<<<gridDim_rem, blockDim_rem, 0,
                                                 stream2>>>(
            oden_array, &offset_ptr[start_rows], iden_ptr, col_ptr,
            &row_ptr[start_rows], nrows_seg, dcols, ((int)dcols / 64) * 64);
        if ((dcols % 32) > 0) {
          cudaStreamCreate(&stream3);
          dim3 gridDim_rem(((int)(nrows_seg - 1) / 8) + 1, 1);
          dim3 blockDim_rem(dcols % 32, 8);
          default_function_kernel_rem_undir<<<gridDim_rem, blockDim_rem, 0,
                                              stream3>>>(
              oden_array, &offset_ptr[start_rows], iden_ptr, col_ptr,
              &row_ptr[start_rows], nrows_seg, dcols,
              (((int)dcols / 64) * 64) + 32);
        }
      } else if ((dcols % 64) > 0) {
        cudaStreamCreate(&stream2);
        dim3 gridDim_rem(((int)(nrows_seg - 1) / 8) + 1, 1);
        dim3 blockDim_rem(dcols % 64, 8);
        default_function_kernel_rem_undir<<<gridDim_rem, blockDim_rem, 0,
                                            stream2>>>(
            oden_array, &offset_ptr[start_rows], iden_ptr, col_ptr,
            &row_ptr[start_rows], nrows_seg, dcols, ((int)dcols / 64) * 64);
      }
    } else {
      if ((int)dcols / 32) {
        cudaStreamCreate(&stream1);
        dim3 gridDim_rem(((int)(nrows_seg - 1) / 8) + 1, (int)dcols / 32);
        dim3 blockDim_rem(32, 8);
        default_function_kernel32_undir<<<gridDim_rem, blockDim_rem, 0,
                                          stream1>>>(
            oden_array, &offset_ptr[start_rows], iden_ptr, col_ptr,
            &row_ptr[start_rows], nrows_seg, dcols);
        if ((dcols % 32) > 0) {
          cudaStreamCreate(&stream2);
          dim3 gridDim_rem(((int)(nrows_seg - 1) / 8) + 1, 1);
          dim3 blockDim_rem(dcols % 32, 8);
          default_function_kernel_rem_undir<<<gridDim_rem, blockDim_rem, 0,
                                              stream2>>>(
              oden_array, &offset_ptr[start_rows], iden_ptr, col_ptr,
              &row_ptr[start_rows], nrows_seg, dcols, (((int)dcols / 32) * 32));
        }
      } else {
        cudaStreamCreate(&stream1);
        dim3 gridDim_rem(((int)(nrows_seg - 1) / 8) + 1, 1);
        dim3 blockDim_rem(dcols % 32, 8);
        default_function_kernel_rem_undir<<<gridDim_rem, blockDim_rem, 0,
                                            stream1>>>(
            oden_array, &offset_ptr[start_rows], iden_ptr, col_ptr,
            &row_ptr[start_rows], nrows_seg, dcols, 0);
      }
    }
    cudaDeviceSynchronize();
  }

  return {output_dense};
}

struct GCN : torch::nn::Module {
  GCN(int in_size, int hidden_size, int out_size, bool dir) {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(in_size, hidden_size));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_size, out_size));
    in_feat_size = in_size;
    hidden_feat_size = hidden_size;
    out_feat_size = out_size;
    directed = dir;
  }

  // Layers should be 602, 32, 50
  // TODOs -- for training aware
  //  DCSR - Having it should count as a format seletion, but to be better it
  //  needs CSC kernels Automatic operation reordering Boolean mask skip nodes
  //  -- Training aware Initial SpMM is in init -- Training aware SENSEi style
  //  of execution -- How do you expose this?? Memory usage optimization --
  //  Based on what Vimarsh said, this comes about because the graph is being
  //  stored multiple times. For an undirected graph, you can directly use the
  //  existing graph without transpose. Kernel optimization -- Coalesed acceses
  //  which the current implmentation has by default
  //
  //
  // Implement the Net's algorithm.
  std::vector<torch::Tensor>
  forward(torch::Tensor input_dense,   // B
          torch::Tensor offset_graph,  // A_sparse_offset
          torch::Tensor rows_graph,    // A_sparse_row_ids
          torch::Tensor columns_graph, // A_sparse_col_ids
          torch::Tensor value_graph,   // A_sparse_values
          torch::Tensor ranges,        // A_sparse_tile_row_bounds
          int nrows, int segments) {

    // return gather_forward_gcn(input_dense, offset_graph, rows_graph,
    //                           columns_graph, value_graph, ranges, nrows,
    //                           segments, directed);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat)
                       .requires_grad(false)
                       .device(torch::kCUDA, 0);
    auto ones = torch::ones({nrows, 1}, options);
    // torch::Tensor degree =
    //     gather_forward_gcn(ones, offset_graph, columns_graph, value_graph,
    //                        bounds, nrows, segments, directed)[0];

    torch::Tensor degree = torch::pow(ones, -1 / 2);

    torch::Tensor norm_input = degree * input_dense;

    torch::Tensor msg_aggr =
        gather_forward_gcn(norm_input, offset_graph, rows_graph,
                              columns_graph, value_graph, ranges, nrows,
                              segments, directed)[0];

    // Delate the norm_input alloc
    norm_input = torch::zeros({1});

    torch::Tensor msg_update = fc1->forward(msg_aggr);

    msg_aggr = torch::zeros({1});

    torch::Tensor norm_out = degree * msg_update;

    msg_update = torch::zeros({1});

    torch::Tensor msg_relu = torch::relu(norm_out);

    norm_out = torch::zeros({1});

    norm_input = degree * msg_relu;

    msg_relu = torch::zeros({1});

    msg_aggr =
        gather_forward_gcn(norm_input, offset_graph, rows_graph,
                              columns_graph, value_graph, ranges, nrows,
                              segments, directed)[0];

    norm_input = torch::zeros({1});

    msg_update = fc2->forward(msg_aggr);

    msg_aggr = torch::zeros({1});

    norm_out = degree * msg_update;

    msg_update = torch::zeros({1});

    msg_relu = torch::relu(norm_out);

    return {msg_relu};
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  int in_feat_size, hidden_feat_size, out_feat_size;
  bool directed;
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

  torch::Tensor total_row_range;
  torch::Tensor total_rows;
  torch::Tensor total_offsets;
  torch::Tensor total_cols;
  torch::Tensor total_vals;

  std::vector<iT> tile_offsets =
      static_ord_col_breakpoints<SM>(&adj, cols_per_tile);

  iT segments = tile_offsets.size() - 1;

  ord_col_tiling_torch_dcsr(tile_offsets,
                            total_row_range, // Only CPU
                            total_rows, // All below are in both CPU and GPU
                            total_offsets, total_cols, total_vals,
                            &adj); // Except this.

  int dcsr_nrows = total_rows.sizes()[0];
  int dcsr_noffsets = total_offsets.sizes()[0];

  std::cout << "New nrows: " << dcsr_nrows << std::endl; 

  iT *row_ptr = total_rows.data_ptr<iT>();
  iT *offset_ptr = total_offsets.data_ptr<iT>();
  iT *col_ptr = total_cols.data_ptr<iT>();
  vT *val_ptr = total_vals.data_ptr<vT>();

  // Init input with random numbers
  DM input_emb;
  input_emb.build(adj.ncols(), emb_size,
                  DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
  for (diT i = 0; i < adj.nrows(); i++) {
    for (dnT j = 0; j < emb_size; j++) {
      input_emb.vals_ptr()[i * emb_size + j] = (dvT)(rand() % 100) / 100;
    }
  }
  input_emb.set_all(1);

  DM out_emb;
  out_emb.build(adj.nrows(), emb_size,
                DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);

  DM out_emb2;
  out_emb2.build(adj.nrows(), emb_size,
                 DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);

  auto wsum_aggr = wsumAgg<val_t, val_t, ind2_t>;

  // Comparison for checking if SpMM works correctlu
  out_emb.set_all(0);
  out_emb2.set_all(0);

  std::cout << adj.nrows() << " " << adj.ncols() << " " << adj.nvals()
            << std::endl;

  // gSpMM(&adj, &input_emb, &out_emb2, wsum_aggr);

  iT *dA_dcsrOffsets, *dA_columns, *dA_rows;
  float *dA_values, *dB;

  // Malloc on GPU
  CUDA_CHECK(
      cudaMalloc((void **)&dA_dcsrOffsets, (dcsr_noffsets) * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_rows, (dcsr_nrows) * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_columns, nvals * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_values, nvals * sizeof(vT)));
  CUDA_CHECK(cudaMalloc((void **)&dB, (nrows * emb_size) * sizeof(vT)));

  // Copy to GPU
  CUDA_CHECK(cudaMemcpy(dA_dcsrOffsets, offset_ptr,
                        (dcsr_noffsets) * sizeof(iT), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_rows, row_ptr, (dcsr_nrows) * sizeof(iT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_columns, col_ptr, nvals * sizeof(iT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_values, val_ptr, nvals * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, input_emb.vals_ptr(),
                        (nrows * emb_size) * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Create Torch tensors
  // The graph inputs
  auto options_cu_int =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, 0);
  torch::Tensor t_offsets =
      torch::from_blob(dA_dcsrOffsets, {dcsr_noffsets}, options_cu_int);
  torch::Tensor t_rows =
      torch::from_blob(dA_rows, {dcsr_nrows}, options_cu_int);
  torch::Tensor t_cols = torch::from_blob(dA_columns, {nvals}, options_cu_int);

  auto options_cu_float =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, 0);
  torch::Tensor t_vals = torch::from_blob(dA_values, {nvals}, options_cu_float);
  // The dense input
  torch::Tensor t_iden =
      torch::from_blob(dB, {nrows, emb_size}, options_cu_float);

  torch::Device device(torch::kCUDA);
  // Create a new Net.

  double start_init, end_init;
  cudaDeviceSynchronize();
  start_init = get_time();
  auto net = std::make_shared<GCN>(emb_size, 32, 50, false);
  cudaDeviceSynchronize();
  end_init = get_time();

  std::cout << "Initialization time: " << end_init - start_init << std::endl;

  net->to(device);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

  double start, end;
  val_t randVal;
  std::vector<double> times_arr;
  for (size_t epoch = 1; epoch <= num_iters; ++epoch) {
    // Reset gradients.
    optimizer.zero_grad();
    // Execute the model on the input data.
    cudaDeviceSynchronize();
    start = get_time();
    torch::Tensor prediction =
        net->forward(t_iden, t_offsets, t_rows, t_cols, t_vals, total_row_range,
                     nrows, segments)[0];

    cudaDeviceSynchronize();
    end = get_time();

    if (epoch >= skip_cache_warmup) {
      times_arr.push_back(end - start);
    }

    // for (int x = 0; x < nrows; x++) {
    //   for (int y = 0; y < emb_size; y++) {
    //     if (prediction[x][y].item<val_t>() !=
    //         out_emb2.vals_ptr()[x * emb_size + y]) {
    //       std::cout << "The results don't match at: " << x << "," << y << ":  "
    //                 << prediction[x][y].item<val_t>() << ", "
    //                 << out_emb2.vals_ptr()[x * emb_size + y] << std::endl;
    //       break;
    //     }
    //   }
    // }
  }

  CUDA_CHECK(cudaFree(dA_dcsrOffsets));
  CUDA_CHECK(cudaFree(dA_values));
  CUDA_CHECK(cudaFree(dA_columns));
  CUDA_CHECK(cudaFree(dA_rows));
  CUDA_CHECK(cudaFree(dB));

  std::cout << calc_mean(times_arr) << "," << calc_std(times_arr) << std::endl;
}