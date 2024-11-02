//
// Created by damitha on 5/12/24.
//
// Define a new Module.
#include "kernels.cu"
#include <algorithm>

typedef int ind1_t;
typedef int ind2_t;
typedef long lab_t;
typedef float val_t;
typedef int mask_load_t;
typedef bool mask_t;

// Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DM;
typedef DenseMatrix<ind1_t, ind2_t, lab_t> DL;
typedef DenseMatrix<ind1_t, ind2_t, mask_load_t> DBL;
typedef DenseMatrix<ind1_t, ind2_t, mask_t> DB;
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM;

torch::Tensor edge_forward_gcn(torch::Tensor input_dense1,
                               torch::Tensor input_dense2,
                               torch::Tensor offset_graph,
                               torch::Tensor columns_graph,
                               torch::Tensor value_graph) {
  auto nrows = offset_graph.numel() - 1;
  auto nvals = columns_graph.numel();
  auto full_iden = input_dense1.numel();
  auto dcols = full_iden / nrows;

  // // Dense
  // Input
  float *iden_ptr1 = input_dense1.data_ptr<float>();
  float *iden_ptr2 = input_dense2.data_ptr<float>();
  // Output
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat)
                     .requires_grad(true)
                     .device(torch::kCUDA, 0);
  auto output_sparse = torch::zeros({nvals}, options);
  float *oden_array = output_sparse.data_ptr<float>();

  // Sparse
  int *offset_ptr = offset_graph.data_ptr<int>();
  int *col_ptr = columns_graph.data_ptr<int>();
  float *val_ptr = value_graph.data_ptr<float>();

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  dim3 gridDim(((int)(nrows - 1) / 8) + 1);
  dim3 blockDim(32, 8);
  // default_function_kernel_sddmm_mult_undir<<<gridDim, blockDim, 0,
  // stream1>>>(
  //     oden_array, offset_ptr, iden_ptr1, iden_ptr2, col_ptr, nrows, dcols);
  int shared_memory_size = dcols * sizeof(float);
  default_function_kernel_sddmm_mult_undir_shared<<<
      gridDim, blockDim, shared_memory_size, stream1>>>(
      oden_array, offset_ptr, iden_ptr1, iden_ptr2, col_ptr, nrows, dcols);
  // dim3 blockDim(1, 8);
  // default_function_kernel_sddvv_plus_nowarp<<<gridDim, blockDim, 0,
  // stream1>>>(
  //     oden_array, offset_ptr, iden_ptr1, iden_ptr2, col_ptr, nrows);

  return output_sparse;
}

torch::Tensor node_spmv_backward_of_sddmm(torch::Tensor offset_graph,
                                          torch::Tensor columns_graph,
                                          torch::Tensor value_graph,
                                          torch::Tensor bounds, int nrows,
                                          int segments, bool is_directed) {
  // Output
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat)
                     .requires_grad(true)
                     .device(torch::kCUDA, 0);
  auto output_dense = torch::zeros({nrows, 1}, options);
  float *oden_array = output_dense.data_ptr<float>();

  // Sparse
  int *offset_ptr = offset_graph.data_ptr<int>();
  int *col_ptr = columns_graph.data_ptr<int>();
  float *val_ptr = value_graph.data_ptr<float>();
  int *bounds_ptr = bounds.data_ptr<int>();

  for (int i = 0; i < segments; i++) {
    int i1 = i;
    int start_vals = bounds_ptr[i1 * 2];
    // int end_vals = bounds_ptr[i1 * 2 + 1];
    // int nvals = end_vals - start_vals;

    cudaStream_t stream1;

    cudaStreamCreate(&stream1);
    dim3 gridDim_rem(((int)(nrows - 1) / 32) + 1);
    dim3 blockDim_rem(32);
    default_function_kernel_spmm_backward_sddmm_32<<<gridDim_rem, blockDim_rem,
                                                     0, stream1>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], val_ptr,
        &col_ptr[start_vals], nrows);
  }

  return output_dense;
}

torch::Tensor inplace_softmax_sddvv(torch::Tensor row_val,
                                    torch::Tensor offset_graph,
                                    torch::Tensor columns_graph,
                                    torch::Tensor value_graph,
                                    torch::Tensor bounds, int nrows,
                                    int segments, bool is_directed) {
  float *row_val_ptr = row_val.data_ptr<float>();
  // Sparse
  int *offset_ptr = offset_graph.data_ptr<int>();
  int *col_ptr = columns_graph.data_ptr<int>();
  float *val_ptr = value_graph.data_ptr<float>();
  int *bounds_ptr = bounds.data_ptr<int>();

  for (int i = 0; i < segments; i++) {
    int i1 = i;
    int start_vals = bounds_ptr[i1 * 2];
    // int end_vals = bounds_ptr[i1 * 2 + 1];
    // int nvals = end_vals - start_vals;

    cudaStream_t stream1;

    cudaStreamCreate(&stream1);
    dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1);
    dim3 blockDim_rem(32, 8);
    default_function_kernel_softmax_sddvv_undir<<<gridDim_rem, blockDim_rem, 0,
                                                  stream1>>>(
        val_ptr, &offset_ptr[i1 * (nrows + 1)], row_val_ptr,
        &col_ptr[start_vals], nrows);
  }

  return value_graph;
}

torch::Tensor inplace_softmax_sddvv_mult(torch::Tensor row_val,
                                         torch::Tensor offset_graph,
                                         torch::Tensor columns_graph,
                                         torch::Tensor value_graph,
                                         torch::Tensor bounds, int nrows,
                                         int segments, bool is_directed) {
  float *row_val_ptr = row_val.data_ptr<float>();
  // Sparse
  int *offset_ptr = offset_graph.data_ptr<int>();
  int *col_ptr = columns_graph.data_ptr<int>();
  float *val_ptr = value_graph.data_ptr<float>();
  int *bounds_ptr = bounds.data_ptr<int>();

  for (int i = 0; i < segments; i++) {
    int i1 = i;
    int start_vals = bounds_ptr[i1 * 2];
    // int end_vals = bounds_ptr[i1 * 2 + 1];
    // int nvals = end_vals - start_vals;

    cudaStream_t stream1;

    cudaStreamCreate(&stream1);
    dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1);
    dim3 blockDim_rem(32, 8);
    default_function_kernel_mult_sddvv_undir<<<gridDim_rem, blockDim_rem, 0,
                                               stream1>>>(
        val_ptr, &offset_ptr[i1 * (nrows + 1)], row_val_ptr,
        &col_ptr[start_vals], nrows);
  }

  return value_graph;
}

torch::Tensor edge_sddvv(torch::Tensor input_dense1, torch::Tensor input_dense2,
                         torch::Tensor offset_graph,
                         torch::Tensor columns_graph, torch::Tensor value_graph,
                         torch::Tensor bounds, int nrows, int segments,
                         bool is_directed) {
  auto nvals = columns_graph.numel();
  // auto full_iden = input_dense1.numel();
  // auto dcols = full_iden / nrows;

  // // Dense
  // Input
  float *iden_ptr1 = input_dense1.data_ptr<float>();
  float *iden_ptr2 = input_dense2.data_ptr<float>();
  // Output
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat)
                     .requires_grad(true)
                     .device(torch::kCUDA, 0);
  auto output_sparse = torch::zeros({nvals}, options);
  float *oden_array = output_sparse.data_ptr<float>();

  // Sparse
  int *offset_ptr = offset_graph.data_ptr<int>();
  int *col_ptr = columns_graph.data_ptr<int>();
  float *val_ptr = value_graph.data_ptr<float>();
  int *bounds_ptr = bounds.data_ptr<int>();

  for (int i = 0; i < segments; i++) {
    int i1 = i;
    int start_vals = bounds_ptr[i1 * 2];
    // int end_vals = bounds_ptr[i1 * 2 + 1];
    // int nvals = end_vals - start_vals;

    cudaStream_t stream1;

    if (is_directed) {
      cudaStreamCreate(&stream1);
      dim3 gridDim(((int)(nrows - 1) / 8) + 1);
      dim3 blockDim(32, 8);
      default_function_kernel_sddvv_plus_undir<<<gridDim, blockDim, 0,
                                                 stream1>>>(
          &oden_array[start_vals], &offset_ptr[i1 * (nrows + 1)], iden_ptr1,
          iden_ptr2, &col_ptr[start_vals], nrows);
    }
  }

  return output_sparse;
}

torch::Tensor edge_sddmm(torch::Tensor input_dense1, torch::Tensor input_dense2,
                         torch::Tensor offset_graph,
                         torch::Tensor columns_graph, torch::Tensor value_graph,
                         torch::Tensor bounds, int nrows, int segments,
                         bool is_directed) {
  auto nvals = columns_graph.numel();
  auto full_iden = input_dense1.numel();
  auto dcols = full_iden / nrows;

  // // Dense
  // Input
  float *iden_ptr1 = input_dense1.data_ptr<float>();
  float *iden_ptr2 = input_dense2.data_ptr<float>();
  // Output
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat)
                     .requires_grad(true)
                     .device(torch::kCUDA, 0);
  auto output_sparse = torch::zeros({nvals}, options);
  float *oden_array = output_sparse.data_ptr<float>();

  // Sparse
  int *offset_ptr = offset_graph.data_ptr<int>();
  int *col_ptr = columns_graph.data_ptr<int>();
  float *val_ptr = value_graph.data_ptr<float>();
  int *bounds_ptr = bounds.data_ptr<int>();

  for (int i = 0; i < segments; i++) {
    int i1 = i;
    int start_vals = bounds_ptr[i1 * 2];
    // int end_vals = bounds_ptr[i1 * 2 + 1];
    // int nvals = end_vals - start_vals;

    cudaStream_t stream1;

    if (is_directed) {
      cudaStreamCreate(&stream1);
      dim3 gridDim(((int)(nrows - 1) / 8) + 1);
      dim3 blockDim(32, 8);
      int shared_memory_size = dcols * sizeof(float);
      default_function_kernel_sddmm_mult_undir_shared<<<
          gridDim, blockDim, shared_memory_size, stream1>>>(
          &oden_array[start_vals], &offset_ptr[i1 * (nrows + 1)], iden_ptr1,
          iden_ptr2, &col_ptr[start_vals], nrows, dcols);
    }
  }

  return output_sparse;
}

// class GatherForward : public torch::autograd::Function<GatherForward> {
// public:
//   static torch::Tensor
//   forward(torch::autograd::AutogradContext *ctx, torch::Tensor input_dense1,
//           torch::Tensor input_dense2, torch::Tensor offset_graph,
//           torch::Tensor columns_graph, torch::Tensor value_graph) {
//     ctx->save_for_backward({offset_graph, columns_graph, value_graph});
//     return edge_forward_gcn(input_dense1, input_dense2, offset_graph,
//                             columns_graph, value_graph);
//   }

//   static torch::autograd::tensor_list
//   backward(torch::autograd::AutogradContext *ctx,
//            torch::autograd::tensor_list grad_outputs) {
//     torch::Tensor input_dense = grad_outputs[0];
//     auto saved = ctx->get_saved_variables();
//     torch::Tensor offset_graph = saved[0];
//     torch::Tensor columns_graph = saved[1];
//     torch::Tensor value_graph = saved[2];
//     return {gather_forward_gcn(input_dense, offset_graph, columns_graph,
//                                value_graph),
//             torch::Tensor(), torch::Tensor(), torch::Tensor()};
//   }
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

  // Implement the Net's algorithm.
  std::vector<torch::Tensor>
  forward(torch::Tensor input_dense1,  // B
          torch::Tensor input_dense2,  // B
          torch::Tensor offset_graph,  // A_sparse_offset
          torch::Tensor columns_graph, // A_sparse_col_ids
          torch::Tensor value_graph,   // A_sparse_values
          int nrows) {
    return {edge_forward_gcn(input_dense1, input_dense2, offset_graph,
                             columns_graph, value_graph)};
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

  // Timing configs
  int num_iters = stoi(string(argv[2]));

  // Column tiling
  iT cols_per_tile = stoi(string(argv[3]));

  // Const settings
  int skip_cache_warmup = 5;

  std::string filename;
  SM adj;
  filename = path;
  readSM_npy32<SM>(filename, &adj);

  std::cout << "PyTorch version: "
    << TORCH_VERSION_MAJOR << "."
    << TORCH_VERSION_MINOR << "."
    << TORCH_VERSION_PATCH << std::endl;
  // Adj info
  
  iT nrows = adj.nrows();
  iT ncols = adj.ncols();
  nT nvals = adj.nvals();

  int emb_size = 32;

  // Init input with random numbers
  DM input_emb1;
  input_emb1.build(nrows, emb_size,
                   DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
  // for (diT i = 0; i < adj.nrows(); i++) {
  //   for (dnT j = 0; j < emb_size; j++) {
  //     input_emb.vals_ptr()[i * emb_size + j] = (dvT)(rand() % 100) / 100;
  //   }
  // }
  input_emb1.set_all(1);

  DM input_emb2;
  input_emb2.build(nrows, emb_size,
                   DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
  // for (diT i = 0; i < adj.nrows(); i++) {
  //   for (dnT j = 0; j < emb_size; j++) {
  //     input_emb.vals_ptr()[i * emb_size + j] = (dvT)(rand() % 100) / 100;
  //   }
  // }
  input_emb2.set_all(1);

  // iT emb_size = input_emb1.ncols();

  torch::Device device(torch::kCUDA);
  // Create a new Net.

  int *dA_csrOffsets, *dA_columns;
  float *dA_values, *dB1, *dB2;

  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets, (nrows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&dA_columns, nvals * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&dA_values, nvals * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&dB1, (nrows * emb_size) * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&dB2, (nrows * emb_size) * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dA_csrOffsets, adj.offset_ptr(),
                        (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_columns, adj.ids_ptr(), nvals * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_values, adj.vals_ptr(), nvals * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB1, input_emb1.vals_ptr(),
                        (nrows * emb_size) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB2, input_emb2.vals_ptr(),
                        (nrows * emb_size) * sizeof(float),
                        cudaMemcpyHostToDevice));
  auto options_cu_int = torch::TensorOptions()
                            .dtype(torch::kInt)
                            .requires_grad(false)
                            .device(torch::kCUDA, 0);
  torch::Tensor t_offsets =
      torch::from_blob(dA_csrOffsets, {nrows + 1}, options_cu_int);
  torch::Tensor t_cols = torch::from_blob(dA_columns, {nvals}, options_cu_int);

  auto options_cu_float_grad = torch::TensorOptions()
                                   .dtype(torch::kFloat)
                                   .requires_grad(true)
                                   .device(torch::kCUDA, 0);
  auto options_cu_float_ngrad = torch::TensorOptions()
                                    .dtype(torch::kFloat)
                                    .requires_grad(false)
                                    .device(torch::kCUDA, 0);
  torch::Tensor t_vals =
      torch::from_blob(dA_values, {nvals}, options_cu_float_ngrad);
  torch::Tensor t_iden1 =
      torch::from_blob(dB1, {nrows, emb_size}, options_cu_float_grad);
  torch::Tensor t_iden2 =
      torch::from_blob(dB2, {nrows, emb_size}, options_cu_float_grad);

  double start_init, end_init;
  cudaDeviceSynchronize();
  start_init = get_time();
  auto net = std::make_shared<GCN>(emb_size, 32, 1, false);
  cudaDeviceSynchronize();
  end_init = get_time();

  std::cout << "Initialization time: " << end_init - start_init << std::endl;

  net->to(device);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::Adam optimizer(
      net->parameters(), torch::optim::AdamOptions(1e-2).weight_decay(5e-4));

  double start, end;
  double start_train, end_train;
  val_t randVal;
  std::vector<double> times_arr, times_arr_train;
  for (size_t epoch = 1; epoch <= num_iters; ++epoch) {
    // Reset gradients.
    optimizer.zero_grad();
    // Execute the model on the input data.
    cudaDeviceSynchronize();
    start = get_time();
    torch::Tensor prediction =
        net->forward(t_iden1, t_iden2, t_offsets, t_cols, t_vals, nrows)[0];

    cudaDeviceSynchronize();
    end = get_time();

    // cudaDeviceSynchronize();
    // start_train = get_time();

    // torch::Tensor prediction_train = prediction.index({t_train_mask});
    // torch::Tensor labels_train = t_labs.index({t_train_mask});

    // auto criterion = torch::nn::CrossEntropyLoss();
    // torch::Tensor d_loss = criterion(prediction_train, labels_train);

    // d_loss.backward();

    // optimizer.step();

    // cudaDeviceSynchronize();
    // end_train = get_time();

    // torch::Tensor prediction_test = prediction.index({t_test_mask});
    // torch::Tensor labels_test = t_labs.index({t_test_mask});

    // auto [pred_val, pred_idx] = torch::max({prediction_test}, 1);

    // auto correct = torch::sum(pred_idx == labels_test);

// #pragma omp parallel for schedule(static)
//     for (int x = 0; x < nvals; x++) {
//       if (prediction[x].item<val_t>() != emb_size) {
//         std::cout << "The results don't match at: " << x << ":  "
//                   << prediction[x].item<val_t>() << std::endl;
//       }
//     }

    // std::cout << "Epoch " << epoch << " Loss: " << d_loss.item<val_t>()
    //           << " Accuracy: "
    //           << (correct.item<val_t>() * 100.0 / labels_test.sizes()[0])
    //           << std::endl;

    if (epoch >= skip_cache_warmup) {
      times_arr.push_back(end - start);
      // times_arr_train.push_back(end_train - start_train);
    }
  }

  CUDA_CHECK(cudaFree(dA_csrOffsets));
  CUDA_CHECK(cudaFree(dA_values));
  CUDA_CHECK(cudaFree(dA_columns));
  CUDA_CHECK(cudaFree(dB1));
  CUDA_CHECK(cudaFree(dB2));

  std::cout << "Inference: " << calc_mean(times_arr) << ","
            << calc_std(times_arr) << std::endl;
  // std::cout << "Train: " << calc_mean(times_arr_train) << ","
  //           << calc_std(times_arr_train) << std::endl;
}