//
// Created by damitha on 5/12/24.
//
// Define a new Module.
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

torch::Tensor gather_forward_gcn(torch::Tensor input_dense,
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
                     .requires_grad(true);
  auto output_dense = torch::zeros({nrows, dcols}, options);
  float *oden_array = output_dense.data_ptr<float>();

  // Sparse
  int *offset_ptr = offset_graph.data_ptr<int>();
  int *col_ptr = columns_graph.data_ptr<int>();
  float *val_ptr = value_graph.data_ptr<float>();

  auto wsum_aggr = wsumAgg<val_t, val_t, ind2_t>;
  // Add SpMM call here
  gSpMM_torch(iden_ptr, offset_ptr, col_ptr, val_ptr, oden_array, (int)nrows, (int)dcols, wsum_aggr);
  return output_dense;
}

class GatherForward : public torch::autograd::Function<GatherForward> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor input_dense,
                               torch::Tensor offset_graph,
                               torch::Tensor columns_graph,
                               torch::Tensor value_graph) {
    ctx->save_for_backward({offset_graph, columns_graph, value_graph});
    return gather_forward_gcn(input_dense, offset_graph, columns_graph,
                              value_graph);
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    torch::Tensor input_dense = grad_outputs[0];
    auto saved = ctx->get_saved_variables();
    torch::Tensor offset_graph = saved[0];
    torch::Tensor columns_graph = saved[1];
    torch::Tensor value_graph = saved[2];
    return {gather_forward_gcn(input_dense, offset_graph, columns_graph,
                               value_graph),
            torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

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
  forward(torch::Tensor input_dense,   // B
          torch::Tensor offset_graph,  // A_sparse_offset
          torch::Tensor columns_graph, // A_sparse_col_ids
          torch::Tensor value_graph,   // A_sparse_values
          int nrows) {
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat)
                       .requires_grad(false);
    auto ones = torch::ones({nrows, 1}, options);
    torch::Tensor degree =
        gather_forward_gcn(ones, offset_graph, columns_graph, value_graph);

    degree = torch::pow(degree, -0.5);

    torch::Tensor res = degree * input_dense;
    res = GatherForward::apply(res, offset_graph, columns_graph, value_graph);
    res = fc1->forward(res);
    res = degree * res;
    res = torch::relu(res);
    res = degree * res;
    res = GatherForward::apply(res, offset_graph, columns_graph, value_graph);
    res = fc2->forward(res);
    res = degree * res;
    return {torch::log_softmax(res, /*dim=*/1)};
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

//  // Column tiling
//  iT cols_per_tile = stoi(string(argv[3]));

  // Const settings
  int skip_cache_warmup = 5;

  std::string filename;
  SM adj;
  filename = path;
  readSM_npy32<SM>(filename, &adj);

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

  std::cout << adj.nrows() << " " << adj.ncols() << " " << adj.nvals()
            << std::endl;
  int classes =
      *std::max_element(labels.vals_ptr(), labels.vals_ptr() + labels.nvals()) +
      1;
  std::cout << "Classes: " << classes << std::endl;
  auto options_cu_int = torch::TensorOptions()
                            .dtype(torch::kInt)
                            .requires_grad(false);
  torch::Tensor t_offsets =
      torch::from_blob(adj.offset_ptr(), {nrows + 1}, options_cu_int);
  torch::Tensor t_cols = torch::from_blob(adj.ids_ptr(), {nvals}, options_cu_int);

  auto options_cu_long =
      torch::TensorOptions().dtype(torch::kLong);
  torch::Tensor t_labs = torch::from_blob(labels.vals_ptr(), {nrows}, options_cu_long);

  auto options_cu_float_grad = torch::TensorOptions()
                                   .dtype(torch::kFloat)
                                   .requires_grad(true);
  auto options_cu_float_ngrad = torch::TensorOptions()
                                    .dtype(torch::kFloat)
                                    .requires_grad(false);
  torch::Tensor t_vals =
      torch::from_blob(adj.vals_ptr(), {nvals}, options_cu_float_ngrad);
  torch::Tensor t_iden =
      torch::from_blob(input_emb.vals_ptr(), {nrows, emb_size}, options_cu_float_grad);

  auto options_cu_bool = torch::TensorOptions()
                             .dtype(torch::kBool)
                             .requires_grad(false);
  torch::Tensor t_train_mask =
      torch::from_blob(train_mask.vals_ptr(), {nrows}, options_cu_bool);
  torch::Tensor t_valid_mask =
      torch::from_blob(valid_mask.vals_ptr(), {nrows}, options_cu_bool);
  torch::Tensor t_test_mask =
      torch::from_blob(test_mask.vals_ptr(), {nrows}, options_cu_bool);

  double start_init, end_init;
  start_init = get_time();
  auto net = std::make_shared<GCN>(emb_size, 32, classes, false);
  end_init = get_time();

  std::cout << "Initialization time: " << end_init - start_init << std::endl;

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
    start = get_time();
    torch::Tensor prediction =
        net->forward(t_iden, t_offsets, t_cols, t_vals, nrows)[0];
    end = get_time();

    start_train = get_time();
    torch::Tensor prediction_train = prediction.index({t_train_mask});
    torch::Tensor labels_train = t_labs.index({t_train_mask});

    auto criterion = torch::nn::CrossEntropyLoss();
    torch::Tensor d_loss = criterion(prediction_train, labels_train);

    d_loss.backward();

    optimizer.step();
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

  std::cout << "Inference: " << calc_mean(times_arr) << ","
            << calc_std(times_arr) << std::endl;
  std::cout << "Train: " << calc_mean(times_arr_train) << ","
            << calc_std(times_arr_train) << std::endl;
}