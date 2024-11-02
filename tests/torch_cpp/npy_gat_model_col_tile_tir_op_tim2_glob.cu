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

int global_nrows;
std::vector<int> global_segments;
bool global_is_directed;

std::vector<torch::Tensor> global_offset_graph;
std::vector<torch::Tensor> global_columns_graph;
std::vector<torch::Tensor> global_value_graph;
std::vector<torch::Tensor> global_bounds;

torch::Tensor gather_forward_gcn(torch::Tensor input_dense,
                                 torch::Tensor offset_graph,
                                 torch::Tensor columns_graph,
                                 torch::Tensor value_graph,
                                 torch::Tensor bounds, int nrows, int segments,
                                 bool is_directed) {
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
  int *bounds_ptr = bounds.data_ptr<int>();

  for (int i = 0; i < segments; i++) {
    int i1 = i;
    int start_vals = bounds_ptr[i1 * 2];
    // int end_vals = bounds_ptr[i1 * 2 + 1];
    // int nvals = end_vals - start_vals;

    cudaStream_t stream1, stream2, stream3;

    if (is_directed) {
      if ((int)dcols / 64) {
        cudaStreamCreate(&stream1);
        dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 64);
        dim3 blockDim(32, 8);
        default_function_kernel64<<<gridDim, blockDim, 0, stream1>>>(
            oden_array, &offset_ptr[i1 * (nrows + 1)], &val_ptr[start_vals],
            iden_ptr, &col_ptr[start_vals], nrows, dcols);

        if ((dcols % 64) > 32) {
          cudaStreamCreate(&stream2);
          dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
          dim3 blockDim_rem(32, 8);
          default_function_kernel32_offset<<<gridDim_rem, blockDim_rem, 0,
                                             stream2>>>(
              oden_array, &offset_ptr[i1 * (nrows + 1)], &val_ptr[start_vals],
              iden_ptr, &col_ptr[start_vals], nrows, dcols,
              ((int)dcols / 64) * 64);
          if ((dcols % 32) > 0) {
            cudaStreamCreate(&stream3);
            dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
            dim3 blockDim_rem(dcols % 32, 8);
            default_function_kernel_rem<<<gridDim_rem, blockDim_rem, 0,
                                          stream3>>>(
                oden_array, &offset_ptr[i1 * (nrows + 1)], &val_ptr[start_vals],
                iden_ptr, &col_ptr[start_vals], nrows, dcols,
                (((int)dcols / 64) * 64) + 32);
          }
        } else if ((dcols % 64) > 0) {
          cudaStreamCreate(&stream2);
          dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
          dim3 blockDim_rem(dcols % 64, 8);
          default_function_kernel_rem<<<gridDim_rem, blockDim_rem, 0,
                                        stream2>>>(
              oden_array, &offset_ptr[i1 * (nrows + 1)], &val_ptr[start_vals],
              iden_ptr, &col_ptr[start_vals], nrows, dcols,
              ((int)dcols / 64) * 64);
        }
      } else {
        if ((int)dcols / 32) {
          cudaStreamCreate(&stream1);
          dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, (int)dcols / 32);
          dim3 blockDim_rem(32, 8);
          default_function_kernel32<<<gridDim_rem, blockDim_rem, 0, stream1>>>(
              oden_array, &offset_ptr[i1 * (nrows + 1)], &val_ptr[start_vals],
              iden_ptr, &col_ptr[start_vals], nrows, dcols);
          if ((dcols % 32) > 0) {
            cudaStreamCreate(&stream2);
            dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
            dim3 blockDim_rem(dcols % 32, 8);
            default_function_kernel_rem<<<gridDim_rem, blockDim_rem, 0,
                                          stream2>>>(
                oden_array, &offset_ptr[i1 * (nrows + 1)], &val_ptr[start_vals],
                iden_ptr, &col_ptr[start_vals], nrows, dcols,
                (((int)dcols / 32) * 32));
          }
        } else {
          cudaStreamCreate(&stream1);
          dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
          dim3 blockDim_rem(dcols % 32, 8);
          default_function_kernel_rem<<<gridDim_rem, blockDim_rem, 0,
                                        stream1>>>(
              oden_array, &offset_ptr[i1 * (nrows + 1)], &val_ptr[start_vals],
              iden_ptr, &col_ptr[start_vals], nrows, dcols, 0);
        }
      }
    } else {
      if ((int)dcols / 64) {
        cudaStreamCreate(&stream1);
        dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 64);
        dim3 blockDim(32, 8);
        default_function_kernel64_undir<<<gridDim, blockDim, 0, stream1>>>(
            oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
            &col_ptr[start_vals], nrows, dcols);

        if ((dcols % 64) > 32) {
          cudaStreamCreate(&stream2);
          dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
          dim3 blockDim_rem(32, 8);
          default_function_kernel32_offset_undir<<<gridDim_rem, blockDim_rem, 0,
                                                   stream2>>>(
              oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
              &col_ptr[start_vals], nrows, dcols, ((int)dcols / 64) * 64);
          if ((dcols % 32) > 0) {
            cudaStreamCreate(&stream3);
            dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
            dim3 blockDim_rem(dcols % 32, 8);
            default_function_kernel_rem_undir<<<gridDim_rem, blockDim_rem, 0,
                                                stream3>>>(
                oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
                &col_ptr[start_vals], nrows, dcols,
                (((int)dcols / 64) * 64) + 32);
          }
        } else if ((dcols % 64) > 0) {
          cudaStreamCreate(&stream2);
          dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
          dim3 blockDim_rem(dcols % 64, 8);
          default_function_kernel_rem_undir<<<gridDim_rem, blockDim_rem, 0,
                                              stream2>>>(
              oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
              &col_ptr[start_vals], nrows, dcols, ((int)dcols / 64) * 64);
        }
      } else {
        if ((int)dcols / 32) {
          cudaStreamCreate(&stream1);
          dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, (int)dcols / 32);
          dim3 blockDim_rem(32, 8);
          default_function_kernel32_undir<<<gridDim_rem, blockDim_rem, 0,
                                            stream1>>>(
              oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
              &col_ptr[start_vals], nrows, dcols);
          if ((dcols % 32) > 0) {
            cudaStreamCreate(&stream2);
            dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
            dim3 blockDim_rem(dcols % 32, 8);
            default_function_kernel_rem_undir<<<gridDim_rem, blockDim_rem, 0,
                                                stream2>>>(
                oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
                &col_ptr[start_vals], nrows, dcols, (((int)dcols / 32) * 32));
          }
        } else {
          cudaStreamCreate(&stream1);
          dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
          dim3 blockDim_rem(dcols % 32, 8);
          default_function_kernel_rem_undir<<<gridDim_rem, blockDim_rem, 0,
                                              stream1>>>(
              oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
              &col_ptr[start_vals], nrows, dcols, 0);
        }
      }
    }
  }

  return output_dense;
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
        oden_array, &offset_ptr[i1 * (nrows + 1)], &val_ptr[start_vals],
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
        &val_ptr[start_vals], &offset_ptr[i1 * (nrows + 1)], row_val_ptr,
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
        &val_ptr[start_vals], &offset_ptr[i1 * (nrows + 1)], row_val_ptr,
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

class NodeAggregate : public torch::autograd::Function<NodeAggregate> {
public:
  static torch::Tensor
  forward(torch::autograd::AutogradContext *ctx, torch::Tensor input_dense,
          torch::Tensor offset_graph, torch::Tensor columns_graph,
          torch::Tensor value_graph, torch::Tensor bounds) {
    ctx->save_for_backward(
        {offset_graph, columns_graph, value_graph, bounds, input_dense});
    return gather_forward_gcn(input_dense, offset_graph, columns_graph,
                              value_graph, bounds, global_nrows,
                              global_segments, global_is_directed);
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    torch::Tensor dZ = grad_outputs[0];
    auto saved = ctx->get_saved_variables();
    torch::Tensor offset_graph = saved[0];
    torch::Tensor columns_graph = saved[1];
    torch::Tensor value_graph = saved[2];
    torch::Tensor bounds = saved[3];
    torch::Tensor X = saved[4];

    // TODO: You can use ctx to see if a certain input requires a grad at all.
    // THEN calcuate the grad without always doing it.

    return {gather_forward_gcn(dZ, offset_graph, columns_graph, value_graph,
                               bounds, global_nrows, global_segments,
                               global_is_directed),
            torch::Tensor(), torch::Tensor(),
            edge_sddmm(dZ, X, offset_graph, columns_graph, value_graph, bounds,
                       global_nrows, global_segments, global_is_directed),
            torch::Tensor()};
  }
};

class EdgeAggregate : public torch::autograd::Function<EdgeAggregate> {
public:
  static torch::Tensor
  forward(torch::autograd::AutogradContext *ctx, torch::Tensor input_dense1,
          torch::Tensor input_dense2, torch::Tensor offset_graph,
          torch::Tensor columns_graph, torch::Tensor value_graph,
          torch::Tensor bounds) {
    ctx->save_for_backward({offset_graph, columns_graph, bounds});
    return edge_sddvv(input_dense1, input_dense2, offset_graph, columns_graph,
                      value_graph, bounds, global_nrows, global_segments,
                      global_is_directed);
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    torch::Tensor d_value_graph = grad_outputs[0];
    auto saved = ctx->get_saved_variables();
    torch::Tensor offset_graph = saved[0];
    torch::Tensor columns_graph = saved[1];
    torch::Tensor bounds = saved[2];
    return {node_spmv_backward_of_sddmm(
                offset_graph, columns_graph, // This should be the reverse graph
                d_value_graph, bounds, global_nrows, global_segments,
                global_is_directed),
            node_spmv_backward_of_sddmm(
                offset_graph,
                columns_graph, // This should be the original graph
                d_value_graph, bounds, global_nrows, global_segments,
                global_is_directed),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()};
  }
};

class EdgeSoftmax : public torch::autograd::Function<EdgeSoftmax> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor offset_graph,
                               torch::Tensor columns_graph,
                               torch::Tensor value_graph,
                               torch::Tensor bounds) {
    torch::Tensor val_exp = torch::exp(value_graph);
    torch::Tensor row_sum = node_spmv_backward_of_sddmm(
        offset_graph, columns_graph, val_exp, bounds, global_nrows,
        global_segments, global_is_directed);
    // std::cout << "D1" << std::endl;
    // for (int i = 0; i < 100; i++){
    //   std::cout << i << ": " << row_sum[i].item<float>() << std::endl;;
    // }
    value_graph = inplace_softmax_sddvv(row_sum, offset_graph, columns_graph,
                                    val_exp, bounds, global_nrows,
                                    global_segments, global_is_directed);
    ctx->save_for_backward({offset_graph, columns_graph, value_graph, bounds});
    return value_graph;
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    torch::Tensor d_value_graph = grad_outputs[0];
    auto saved = ctx->get_saved_variables();
    torch::Tensor offset_graph = saved[0];
    torch::Tensor columns_graph = saved[1];
    torch::Tensor value_graph = saved[2]; // n x 1
    torch::Tensor bounds = saved[3];

    // std::cout << "Before back" << std::endl;
    // torch::Tensor nan_mask = torch::isnan(d_value_graph);
    // torch::Tensor nan_indices = nan_mask.nonzero();
    // for (int i = 0; i < std::min((int)nan_indices.size(0), 10); ++i) {
    //     auto index = nan_indices[i];
    //     std::cout << "Nan at index (" << index[0].item<int>() << ", " << index[1].item<int>() << "): "
    //               << d_value_graph.index({index[0], index[1]}).item<float>() << std::endl;
    // }

    torch::Tensor sds = value_graph * d_value_graph; // e x 1

    torch::Tensor accum = node_spmv_backward_of_sddmm(
        offset_graph, columns_graph, sds, bounds, global_nrows, global_segments,
        global_is_directed); // n x 1

    torch::Tensor res = inplace_softmax_sddvv_mult(
        accum, offset_graph, columns_graph, value_graph, bounds, global_nrows,
        global_segments, global_is_directed);

    res = sds - res;

    // std::cout << "After back" << std::endl;
    // nan_mask = torch::isnan(res);
    // nan_indices = nan_mask.nonzero();
    // for (int i = 0; i < std::min((int)nan_indices.size(0), 10); ++i) {
    //     auto index = nan_indices[i];
    //     std::cout << "Nan at index (" << index[0].item<int>() << ", " << index[1].item<int>() << "): "
    //               << res.index({index[0], index[1]}).item<float>() << std::endl;
    // }
    // res = d_value_graph;

    return {torch::Tensor(), torch::Tensor(), res, torch::Tensor()};
  }
};

struct GAT : torch::nn::Module {
  GAT(int in_size, int hidden_size, int out_size, bool dir) {
    // Construct and register two Linear submodules.

    fc1 = register_module("fc1", torch::nn::Linear(in_size, hidden_size));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_size, out_size));

    al1 = register_module(
        "al1", torch::nn::Linear(
                   torch::nn::LinearOptions(hidden_size, 1).bias(false)));
    ar1 = register_module(
        "ar1", torch::nn::Linear(
                   torch::nn::LinearOptions(hidden_size, 1).bias(false)));
    al2 = register_module(
        "al2",
        torch::nn::Linear(torch::nn::LinearOptions(out_size, 1).bias(false)));
    ar2 = register_module(
        "ar2",
        torch::nn::Linear(torch::nn::LinearOptions(out_size, 1).bias(false)));

    in_feat_size = in_size;
    hidden_feat_size = hidden_size;
    out_feat_size = out_size;
    directed = dir;
    // This will be confusing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    global_is_directed = true;

    apply_xavier_initialization();
  }

  void apply_xavier_initialization() {
    // Xavier uniform initialization for Linear layers
    torch::NoGradGuard no_grad;  // Disable gradient tracking for initialization
    for (auto& module : modules(/*include_self=*/false)) {
      if (auto* linear = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
        // Apply Xavier uniform initialization to weights
        torch::nn::init::xavier_uniform_(linear->weight);
        // torch::nn::init::kaiming_uniform_(linear->weight);

        // You may also want to initialize the biases to zero (optional)
        if (linear->bias.defined()) {
            torch::nn::init::zeros_(linear->bias);
        }
      }
    }
  }

  std::vector<torch::Tensor>
  forward(torch::Tensor input_dense,   // B
          torch::Tensor offset_graph,  // A_sparse_offset
          torch::Tensor columns_graph, // A_sparse_col_ids
          torch::Tensor value_graph,   // A_sparse_values
          torch::Tensor bounds,        // A_sparse_tile_bounds
          int nrows, int segments, int slope) {
    global_nrows = nrows;
    global_segments = segments;
    torch::nn::LeakyReLU leaky_relu(
        torch::nn::LeakyReLUOptions().negative_slope(slope));

    torch::Tensor res = fc1->forward(input_dense);

    // Edge-level attention calculation
    torch::Tensor attn_l = al1->forward(res);
    torch::Tensor attn_r = ar1->forward(res);

    // torch::Tensor attn = value_graph;
    torch::Tensor attn = EdgeAggregate::apply(attn_l, attn_r, offset_graph, columns_graph,
                                value_graph, bounds);
    // std::cout << "A" << std::endl;
    // for (int i = 0; i < 100; i++){
    //   int j_s = offset_graph[i].item<int>();
    //   int j_e = offset_graph[i + 1].item<int>();
    //   std::cout << i << ": ";
    //   for (int j = j_s; j < j_e; j++){
    //     std::cout << attn[j].item<float>() <<  ",";
    //   }
    //   std::cout << std::endl;
    // }
    attn = leaky_relu->forward(attn);
    // std::cout << "B" << std::endl;
    // for (int i = 0; i < 100; i++){
    //   int j_s = offset_graph[i].item<int>();
    //   int j_e = offset_graph[i + 1].item<int>();
    //   std::cout << i << ": ";
    //   for (int j = j_s; j < j_e; j++){
    //     std::cout << attn[j].item<float>() <<  ",";
    //   }
    //   std::cout << std::endl;
    // }
    attn = EdgeSoftmax::apply(offset_graph, columns_graph, attn, bounds);
    // std::cout << "D" << std::endl;
    // for (int i = 0; i < 100; i++){
    //   int j_s = offset_graph[i].item<int>();
    //   int j_e = offset_graph[i + 1].item<int>();
    //   std::cout << i << ": ";
    //   for (int j = j_s; j < j_e; j++){
    //     std::cout << attn[j].item<float>() <<  ",";
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << attn << std::endl;
    attn = torch::clamp(attn, 1e-7, 0.5);  
    res = NodeAggregate::apply(res, offset_graph, columns_graph, attn, bounds);

    // std::cout << "Before nodes" << std::endl;
    // torch::Tensor inf_mask1 = torch::isinf(res);
    // torch::Tensor inf_indices1 = inf_mask1.nonzero();
    // for (int i = 0; i < std::min((int)inf_indices1.size(0), 10); ++i) {
    //     auto index = inf_indices1[i];
    //     std::cout << "Inf at index (" << index[0].item<int>() << ", " << index[1].item<int>() << "): "
    //               << res.index({index[0], index[1]}).item<float>() << std::endl;
    // }
    // torch::Tensor nan_mask1 = torch::isnan(res);
    // torch::Tensor nan_indices1 = nan_mask1.nonzero();
    // for (int i = 0; i < std::min((int)nan_indices1.size(0), 10); ++i) {
    //     auto index = nan_indices1[i];
    //     std::cout << "Nan at index (" << index[0].item<int>() << ", " << index[1].item<int>() << "): "
    //               << res.index({index[0], index[1]}).item<float>() << std::endl;
    // }

    res = torch::relu(res);

    res = fc2->forward(res);
    // Edge-level attention calculation
    attn_l = al2->forward(res);
    attn_r = ar2->forward(res);
    attn = EdgeAggregate::apply(attn_l, attn_r, offset_graph, columns_graph,
                                attn, bounds);
    // std::cout << "A2" << std::endl;
    // for (int i = 0; i < 100; i++){
    //   int j_s = offset_graph[i].item<int>();
    //   int j_e = offset_graph[i + 1].item<int>();
    //   std::cout << i << ": ";
    //   for (int j = j_s; j < j_e; j++){
    //     std::cout << attn[j].item<float>() <<  ",";
    //   }
    //   std::cout << std::endl;
    // }
    attn = leaky_relu->forward(attn);
    // std::cout << "C2" << std::endl;
    // for (int i = 0; i < 100; i++){
    //   int j_s = offset_graph[i].item<int>();
    //   int j_e = offset_graph[i + 1].item<int>();
    //   std::cout << i << ": ";
    //   for (int j = j_s; j < j_e; j++){
    //     std::cout << attn[j].item<float>() <<  ",";
    //   }
    //   std::cout << std::endl;
    // }
    attn = EdgeSoftmax::apply(offset_graph, columns_graph, attn, bounds);
    attn = torch::clamp(attn, 1e-7, 0.5);  
    // std::cout << "D2" << std::endl;
    // for (int i = 0; i < 100; i++){
    //   int j_s = offset_graph[i].item<int>();
    //   int j_e = offset_graph[i + 1].item<int>();
    //   std::cout << i << ": ";
    //   for (int j = j_s; j < j_e; j++){
    //     std::cout << attn[j].item<float>() <<  ",";
    //   }
    //   std::cout << std::endl;
    // }

    res = NodeAggregate::apply(res, offset_graph, columns_graph, attn,
                               bounds);
    // std::cout << "In the end" << std::endl;
    // torch::Tensor inf_mask = torch::isinf(res);
    // torch::Tensor inf_indices = inf_mask.nonzero();
    // for (int i = 0; i < std::min((int)inf_indices.size(0), 10); ++i) {
    //     auto index = inf_indices[i];
    //     std::cout << "Inf at index (" << index[0].item<int>() << ", " << index[1].item<int>() << "): "
    //               << res.index({index[0], index[1]}).item<float>() << std::endl;
    // }
    // torch::Tensor nan_mask = torch::isnan(res);
    // torch::Tensor nan_indices = nan_mask.nonzero();
    // for (int i = 0; i < std::min((int)nan_indices.size(0), 10); ++i) {
    //     auto index = nan_indices[i];
    //     std::cout << "Nan at index (" << index[0].item<int>() << ", " << index[1].item<int>() << "): "
    //               << res.index({index[0], index[1]}).item<float>() << std::endl;
    // }



    return {torch::log_softmax(res, /*dim=*/1)};
    // return {res};
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  torch::nn::Linear al1{nullptr}, ar1{nullptr}, al2{nullptr}, ar2{nullptr};
  // torch::Tensor al1{nullptr}, ar1{nullptr}, al2{nullptr}, ar2{nullptr};
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

  // bool do_reorder = stoi(string(argv[4]));

  // Const settings
  int skip_cache_warmup = 5;

  std::string filename;
  SM adj;
  filename = path;
  readSM_npy32<SM>(filename, &adj);

  adj.set_all(1);

  // Adj info
  iT nrows = adj.nrows();
  iT ncols = adj.ncols();
  nT nvals = adj.nvals();

  // Init input with random numbers
  DM input_emb;
  readDM_npy<DM>(filename + "Feat.npy", &input_emb,
                 DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
  iT emb_size = input_emb.ncols();

  DL labels;
  readDM_npy<DL>(filename + "Lab.npy", &labels,
                 DenseMatrix<ind1_t, ind2_t, lab_t>::DENSE_MTX_TYPE::RM);

  DBL train_mask_load;
  readDM_npy<DBL>(filename + "TnMsk.npy", &train_mask_load,
                  DBL::DENSE_MTX_TYPE::RM);
  DBL valid_mask_load;
  readDM_npy<DBL>(filename + "VlMsk.npy", &valid_mask_load,
                  DBL::DENSE_MTX_TYPE::RM);
  DBL test_mask_load;
  readDM_npy<DBL>(filename + "TsMsk.npy", &test_mask_load,
                  DBL::DENSE_MTX_TYPE::RM);

  DB train_mask;
  repopulate<DBL, DB>(&train_mask_load, &train_mask);
  DB valid_mask;
  repopulate<DBL, DB>(&valid_mask_load, &valid_mask);
  DB test_mask;
  repopulate<DBL, DB>(&test_mask_load, &test_mask);

  std::vector<SM *> tiled_adj;
  tiled_adj.push_back(&adj);

  torch::Tensor total_offsets;
  torch::Tensor total_cols;
  torch::Tensor total_vals;
  torch::Tensor total_bounds;

  std::vector<iT> tile_offsets =
      static_ord_col_breakpoints<SM>(&adj, cols_per_tile);

  iT segments = tile_offsets.size() - 1;

  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).requires_grad(false);
  auto options_float =
      torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);

  // The first and last value of this should also give the offsets for the
  // columns and vals
  total_offsets = torch::zeros({(adj.nrows() + 1) * (segments)}, options_int);
  total_cols = torch::zeros({adj.nvals()}, options_int);
  total_vals = torch::zeros({adj.nvals()}, options_float);

  total_bounds = torch::zeros({2 * (segments)}, options_int);

  ord_col_tiling_torch(tile_offsets, total_offsets, total_cols, total_vals,
                       total_bounds, &adj);

  iT *offset_ptr = total_offsets.data_ptr<iT>();
  iT *col_ptr = total_cols.data_ptr<iT>();
  vT *val_ptr = total_vals.data_ptr<vT>();

  std::cout << adj.nrows() << " " << adj.ncols() << " " << adj.nvals()
            << std::endl;
  int classes =
      *std::max_element(labels.vals_ptr(), labels.vals_ptr() + labels.nvals()) +
      1;
  std::cout << "Classes: " << classes << std::endl;
  torch::Device device(torch::kCUDA);

  iT *dA_csrOffsets, *dA_columns, *dL;
  float *dA_values, *dB;
  bool *d_train_mask, *d_valid_mask, *d_test_mask;

  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets,
                        ((nrows + 1) * segments) * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_columns, nvals * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_values, nvals * sizeof(vT)));
  CUDA_CHECK(cudaMalloc((void **)&dB, (nrows * emb_size) * sizeof(vT)));
  CUDA_CHECK(cudaMalloc((void **)&dL, nrows * sizeof(long)));
  CUDA_CHECK(cudaMalloc((void **)&d_train_mask, nrows * sizeof(bool)));
  CUDA_CHECK(cudaMalloc((void **)&d_valid_mask, nrows * sizeof(bool)));
  CUDA_CHECK(cudaMalloc((void **)&d_test_mask, nrows * sizeof(bool)));

  CUDA_CHECK(cudaMemcpy(dA_csrOffsets, offset_ptr,
                        ((nrows + 1) * segments) * sizeof(iT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_columns, col_ptr, nvals * sizeof(iT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_values, val_ptr, nvals * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, input_emb.vals_ptr(),
                        (nrows * emb_size) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dL, labels.vals_ptr(), nrows * sizeof(long),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_train_mask, train_mask.vals_ptr(),
                        nrows * sizeof(bool), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_valid_mask, valid_mask.vals_ptr(),
                        nrows * sizeof(bool), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_test_mask, test_mask.vals_ptr(), nrows * sizeof(bool),
                        cudaMemcpyHostToDevice));

  auto options_cu_int =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, 0);
  torch::Tensor t_offsets =
      torch::from_blob(dA_csrOffsets, {(nrows + 1) * segments}, options_cu_int);
  torch::Tensor t_cols = torch::from_blob(dA_columns, {nvals}, options_cu_int);

  auto options_cu_long =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA, 0);
  torch::Tensor t_labs = torch::from_blob(dL, {nrows}, options_cu_long);

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
  torch::Tensor t_iden =
      torch::from_blob(dB, {nrows, emb_size}, options_cu_float_grad);

  auto options_cu_bool = torch::TensorOptions()
                             .dtype(torch::kBool)
                             .requires_grad(false)
                             .device(torch::kCUDA, 0);
  torch::Tensor t_train_mask =
      torch::from_blob(d_train_mask, {nrows}, options_cu_bool);
  torch::Tensor t_valid_mask =
      torch::from_blob(d_valid_mask, {nrows}, options_cu_bool);
  torch::Tensor t_test_mask =
      torch::from_blob(d_test_mask, {nrows}, options_cu_bool);

  double start_init, end_init;
  cudaDeviceSynchronize();
  start_init = get_time();
  auto net = std::make_shared<GAT>(emb_size, 32, classes, false);
  cudaDeviceSynchronize();
  end_init = get_time();

  std::cout << "Initialization time: " << end_init - start_init << std::endl;

  net->to(device);
  // torch::nn::init::xavier_uniform_(net->parameters());
  torch::nn::utils::clip_grad_norm_(net->parameters(), 0.01);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::Adam optimizer(
      net->parameters(), torch::optim::AdamOptions(1e-2).weight_decay(5e-4));

  double start, end;
  double start_train, end_train;
  // val_t randVal;
  std::vector<double> times_arr, times_arr_train;
  for (size_t epoch = 1; epoch <= num_iters; ++epoch) {
    // Reset gradients.
    optimizer.zero_grad();
    // Execute the model on the input data.
    cudaDeviceSynchronize();
    start = get_time();
    torch::Tensor prediction =
        net->forward(t_iden, t_offsets, t_cols, t_vals, total_bounds, nrows,
                     segments, 0.2)[0];

    cudaDeviceSynchronize();
    end = get_time();

    cudaDeviceSynchronize();
    start_train = get_time();

    torch::Tensor prediction_train = prediction.index({t_train_mask});
    torch::Tensor labels_train = t_labs.index({t_train_mask});

    auto criterion = torch::nn::CrossEntropyLoss();
    torch::Tensor d_loss = criterion(prediction_train, labels_train);

    d_loss.backward();

    optimizer.step();

    cudaDeviceSynchronize();
    end_train = get_time();

    torch::Tensor prediction_test = prediction.index({t_test_mask});
    torch::Tensor labels_test = t_labs.index({t_test_mask});

    auto [pred_val, pred_idx] = torch::max({prediction_test}, 1);

    auto correct = torch::sum(pred_idx == labels_test);

    std::cout << "Epoch " << epoch << " Loss: " << d_loss.item<val_t>()
              << " Accuracy: "
              << (correct.item<val_t>() * 100.0 / labels_test.sizes()[0])
              << std::endl;

    if (epoch >= skip_cache_warmup) {
      times_arr.push_back(end - start);
      times_arr_train.push_back(end_train - start_train);
    }
  }

  CUDA_CHECK(cudaFree(dA_csrOffsets));
  CUDA_CHECK(cudaFree(dA_values));
  CUDA_CHECK(cudaFree(dA_columns));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dL));
  CUDA_CHECK(cudaFree(d_train_mask));
  CUDA_CHECK(cudaFree(d_valid_mask));
  CUDA_CHECK(cudaFree(d_test_mask));

  std::cout << "Inference: " << calc_mean(times_arr) << ","
            << calc_std(times_arr) << std::endl;
  std::cout << "Train: " << calc_mean(times_arr_train) << ","
            << calc_std(times_arr_train) << std::endl;
}