//
// Created by damitha on 5/12/24.
//
// Define a new Module.
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>

#include <cmath>
#include <iostream>
#include <parallel/algorithm>
#include <torch/script.h>
#include <vector>

// #include <ATen/ParallelOpenMP.h>
#include <bits/stdc++.h>
#include <omp.h>
#include <stdlib.h>

typedef int ind1_t;
typedef int ind2_t;
typedef float val_t;

#include "../../src/matrix/csrc_matrix.h"
#include "../../src/matrix/dense_matrix.h"
#include "../../src/ops/aggregators.h"
#include "../../src/ops/sparse_matrix_ops.h"
#include "../../src/utils/mtx_io.h"
#include "../common.h"

#include <torch/torch.h>

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

std::vector<at::Tensor> gather_forward_gcn(torch::Tensor input_dense,
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
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat)
                     .requires_grad(true)
                     .device(torch::kCUDA, 0);
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
  CUSPARSE_CHECK(cusparseCreateCsr(&matA, nrows, nrows, nvals, offset_ptr,
                                   col_ptr, val_ptr, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, // Need to change these
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  // Create dense matrix B
  CUSPARSE_CHECK(cusparseCreateDnMat(&matB, nrows, dcols, dcols, iden_ptr,
                                     CUDA_R_32F,
                                     CUSPARSE_ORDER_ROW)); // changed
  // Create dense matrix C
  CUSPARSE_CHECK(cusparseCreateDnMat(&matC, nrows, dcols, dcols, oden_array,
                                     CUDA_R_32F,
                                     CUSPARSE_ORDER_ROW)); // changed

  // allocate an external buffer if needed
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &bufferSize));
  CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

  CUSPARSE_CHECK(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                              matB, &beta, matC, CUDA_R_32F,
                              CUSPARSE_SPMM_CSR_ALG2, dBuffer));

  CUSPARSE_CHECK(cusparseDestroySpMat(matA));
  CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
  CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
  CUSPARSE_CHECK(cusparseDestroy(handle));
  CUDA_CHECK(cudaFree(dBuffer));

  return {output_dense};
}

// struct GCN : torch::nn::Module {

//   // Implement the Net's algorithm.
//   std::vector<torch::Tensor> forward(torch::Tensor input_dense,
//                                      torch::Tensor offset_graph,
//                                      torch::Tensor columns_graph,
//                                      torch::Tensor value_graph) {
//     return gather_forward_gcn(input_dense, offset_graph, columns_graph,
//                               value_graph);

//     //        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
//     //        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
//     //        x = torch::relu(fc2->forward(x));
//     //        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
//     //        return x;
//   }

//   // Use one of many "standard library" modules.
//   torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
// };

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
          torch::Tensor columns_graph, // A_sparse_col_ids
          torch::Tensor value_graph,   // A_sparse_values
          int nrows) {

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat)
                       .requires_grad(false)
                       .device(torch::kCUDA, 0);
    auto ones = torch::ones({nrows, 1}, options);
    torch::Tensor degree =
        gather_forward_gcn(ones, offset_graph, columns_graph, value_graph)[0];

    degree = torch::pow(degree, -1 / 2);

    torch::Tensor norm_input = degree * input_dense;

    torch::Tensor msg_aggr =
        gather_forward_gcn(norm_input, offset_graph, columns_graph, value_graph)[0];

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
        gather_forward_gcn(norm_input, offset_graph, columns_graph, value_graph)[0];

    norm_input = torch::zeros({1});

    msg_update = fc2->forward(msg_aggr);

    msg_aggr = torch::zeros({1});

    norm_out = degree * msg_update;

    msg_update = torch::zeros({1});

    msg_relu = torch::relu(norm_out);

    return {msg_relu};

    // return gather_forward_gcn(input_dense, offset_graph, columns_graph,
    //                           value_graph, bounds, nrows, segments,
    //                           directed);
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

  // gSpMM(&adj, &input_emb, &out_emb2, wsum_aggr);

  std::cout << adj.nrows() << " " << adj.ncols() << " " << adj.nvals()
            << std::endl;

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

  int *dA_csrOffsets, *dA_columns;
  float *dA_values, *dB;

  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets, (nrows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&dA_columns, nvals * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&dA_values, nvals * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&dB, (nrows * emb_size) * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dA_csrOffsets, adj.offset_ptr(),
                        (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_columns, adj.ids_ptr(), nvals * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_values, adj.vals_ptr(), nvals * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, input_emb.vals_ptr(),
                        (nrows * emb_size) * sizeof(float),
                        cudaMemcpyHostToDevice));

  auto options_cu_int =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, 0);
  torch::Tensor t_offsets =
      torch::from_blob(dA_csrOffsets, {nrows + 1}, options_cu_int);
  torch::Tensor t_cols = torch::from_blob(dA_columns, {nvals}, options_cu_int);

  auto options_cu_float =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, 0);
  torch::Tensor t_vals = torch::from_blob(dA_values, {nvals}, options_cu_float);
  torch::Tensor t_iden =
      torch::from_blob(dB, {nrows * emb_size}, options_cu_float);

  double start, end;
  val_t randVal;
  std::vector<double> times_arr;
  for (size_t epoch = 1; epoch <= num_iters; ++epoch) {
    // Reset gradients.
    optimizer.zero_grad();
    // Execute the model on the input data.
    cudaDeviceSynchronize();
    start = get_time();
    //        std::vector<torch::Tensor> prediction = net->forward(t_iden,
    //        t_offsets, t_cols, t_vals);
    torch::Tensor prediction =
        net->forward(t_iden, t_offsets, t_cols, t_vals, nrows)[0];

    cudaDeviceSynchronize();
    end = get_time();

    if (epoch >= skip_cache_warmup) {
      times_arr.push_back(end - start);
    }

    randVal = prediction[nrows - 1][emb_size - 1].item<val_t>();

    //        for (int x = 0; x < nrows; x++){
    //            for (int y = 0; y < emb_size; y++){
    //                if (prediction[x][y].item<val_t>()!= out_emb2.vals_ptr()[x
    //                * emb_size + y]) {
    //                    std::cout << "The results don't match at: " << x <<
    //                    "," << y << ":  " << prediction[x][y].item<val_t>() <<
    //                    ", "
    //                              << out_emb2.vals_ptr()[x * emb_size + y] <<
    //                              std::endl;
    //                    break;
    //                }
    //            }
    //        }

    // Compute a loss value to judge the prediction of our model.
    //        torch::Tensor loss = torch::nll_loss(prediction, batch.target);
    //        // Compute gradients of the loss w.r.t. the parameters of our
    //        model. loss.backward();
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

  std::cout << calc_mean(times_arr) << "," << calc_std(times_arr) << "|"
            << randVal << std::endl;
}