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
  // GCN(int in_size, int hidden_size, int out_size, bool dir) {
  //   // Construct and register two Linear submodules.
  //   fc1 = register_module("fc1", torch::nn::Linear(in_size, hidden_size));
  //   fc2 = register_module("fc2", torch::nn::Linear(hidden_size, out_size));
  //   in_feat_size = in_size;
  //   hidden_feat_size = hidden_size;
  //   out_feat_size = out_size;
  //   directed = dir;
  // }

  GCN(int in_size, int hidden_size, int out_size, bool dir, torch::Tensor w0, torch::Tensor w1) {
    // Construct and register two Linear submodules.
    auto w0_val = torch::nn::Linear(in_size, hidden_size);
    std::cout << "wo: " << w0_val->weight.sizes()[0] << " " << w0_val->weight.sizes()[1] << std::endl;
    w0_val->weight.set_data(w0);
    std::cout << "wo: " << w0_val->weight.sizes()[0] << " " << w0_val->weight.sizes()[1] << std::endl;
    auto w1_val = torch::nn::Linear(hidden_size, out_size);
    w1_val->weight.set_data(w1);
    // fc1 = register_module("fc1", torch::nn::Linear(in_size, hidden_size));
    // fc2 = register_module("fc2", torch::nn::Linear(hidden_size, out_size));

    fc1 = register_module("fc1", w0_val);
    fc2 = register_module("fc2", w1_val);
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
        gather_forward_gcn(ones, offset_graph, columns_graph, value_graph);

    degree = torch::pow(degree, -0.5);
    degree = degree.view({degree.sizes()[0], 1});

    // std::cout << "degree: " << degree.requires_grad() << std::endl;
    std::cout << "degree: " << degree[0][0].item<float>() << " " << degree[1][0].item<float>() << " " << degree[2][0].item<float>() << std::endl;

    std::cout << "input_dense: " << input_dense[0][0].item<float>() << " " << input_dense[0][1].item<float>() << " " << input_dense[1][0].item<float>() << std::endl;
    // std::cout << "input_dense: " << input_dense.requires_grad() << std::endl;

    // torch::Tensor res = degree * input_dense;
    torch::Tensor res = input_dense * degree;

    std::cout << "after_norm: " << res[0][0].item<float>() << " " << res[0][1].item<float>() << " " << res[1][0].item<float>() << std::endl;

    // std::cout << "norm_input: " << norm_input.requires_grad() << std::endl;

    // torch::Tensor msg_aggr = gather_forward_gcn(norm_input, offset_graph,
    //                                             columns_graph, value_graph);
    res = fc1->forward(res);

    std::cout << "matmul: " << res[0][0].item<float>() << " " << res[0][1].item<float>() << " " << res[1][0].item<float>() << std::endl;


    res = GatherForward::apply(res, offset_graph, columns_graph, value_graph);

    std::cout << "aggr: " << res[0][0].item<float>() << " " << res[0][1].item<float>() << " " << res[1][0].item<float>() << std::endl;

    // std::cout << "msg_aggr: " << msg_aggr.requires_grad() << std::endl;

    // Delate the norm_input alloc
    // norm_input = torch::zeros({1});

    

    // std::cout << "msg_update: " << msg_update.requires_grad() << std::endl;

    // msg_aggr = torch::zeros({1});

    // res = degree * res;
    res = res * degree;

    // std::cout << "norm_out: " << norm_out.requires_grad() << std::endl;

    // msg_update = torch::zeros({1});

    res = torch::relu(res);

    // std::cout << "msg_relu: " << msg_relu.requires_grad() << std::endl;

    // norm_out = torch::zeros({1});

    // res = degree * res;
    res = res * degree;

    // std::cout << "norm_input: " << norm_input.requires_grad() << std::endl;

    // msg_relu = torch::zeros({1});

    res = GatherForward::apply(res, offset_graph, columns_graph, value_graph);
    // msg_aggr = GatherForward::apply(norm_input, offset_graph, columns_graph,
    //                                 value_graph);

    // std::cout << "msg_aggr: " << msg_aggr.requires_grad() << std::endl;

    // norm_input = torch::zeros({1});

    res = fc2->forward(res);

    // std::cout << "msg_update: " << msg_update.requires_grad() << std::endl;

    // msg_aggr = torch::zeros({1});

    // res = degree * res;
    res = res * degree;

    // No ReLU in the final layer
    return {torch::log_softmax(res, /*dim=*/1)};

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
  DM w0;
  readDM_npy<DM>(filename + "W0train.npy", &w0,
                 DM::DENSE_MTX_TYPE::RM);
  DM w1;
  readDM_npy<DM>(filename + "W1train.npy", &w1,
                 DM::DENSE_MTX_TYPE::RM);

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

  torch::Device device(torch::kCUDA);
  // Create a new Net.

  int *dA_csrOffsets, *dA_columns, *dL;
  float *dA_values, *dB, *dw0, *dw1;
  bool *d_train_mask, *d_valid_mask, *d_test_mask;

  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets, (nrows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&dA_columns, nvals * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&dA_values, nvals * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&dB, (nrows * emb_size) * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&dL, nrows * sizeof(long)));
  CUDA_CHECK(cudaMalloc((void **)&d_train_mask, nrows * sizeof(bool)));
  CUDA_CHECK(cudaMalloc((void **)&d_valid_mask, nrows * sizeof(bool)));
  CUDA_CHECK(cudaMalloc((void **)&d_test_mask, nrows * sizeof(bool)));
  CUDA_CHECK(cudaMalloc((void **)&dw0, (w0.nrows() * w0.ncols()) * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&dw1, (w1.nrows() * w1.ncols()) * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dA_csrOffsets, adj.offset_ptr(),
                        (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_columns, adj.ids_ptr(), nvals * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_values, adj.vals_ptr(), nvals * sizeof(float),
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
  CUDA_CHECK(cudaMemcpy(dw0, w0.vals_ptr(),
                        (w0.nrows() * w0.ncols()) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw1, w1.vals_ptr(),
                        (w1.nrows() * w1.ncols()) * sizeof(float),
                        cudaMemcpyHostToDevice));

  auto options_cu_int = torch::TensorOptions()
                            .dtype(torch::kInt)
                            .requires_grad(false)
                            .device(torch::kCUDA, 0);
  torch::Tensor t_offsets =
      torch::from_blob(dA_csrOffsets, {nrows + 1}, options_cu_int);
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
  torch::Tensor t_w0 =
      torch::from_blob(dw0, {w0.nrows(), w0.ncols()}, options_cu_float_grad);
  torch::Tensor t_w1 =
      torch::from_blob(dw1, {w1.nrows(), w1.ncols()}, options_cu_float_grad);

  t_w0 = t_w0.transpose(0, 1);
  t_w1 = t_w1.transpose(0, 1);

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
  // auto net = std::make_shared<GCN>(emb_size, 32, classes, false);
  auto net = std::make_shared<GCN>(emb_size, 32, classes, false, t_w0, t_w1);
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
    //        std::vector<torch::Tensor> prediction = net->forward(t_iden,
    //        t_offsets, t_cols, t_vals);
    torch::Tensor prediction =
        net->forward(t_iden, t_offsets, t_cols, t_vals, nrows)[0];

    cudaDeviceSynchronize();
    end = get_time();

    cudaDeviceSynchronize();
    start_train = get_time();

    torch::Tensor prediction_train = prediction.index({t_train_mask});
    torch::Tensor labels_train = t_labs.index({t_train_mask});

    torch::Tensor prediction_test = prediction.index({t_test_mask});
    torch::Tensor labels_test = t_labs.index({t_test_mask});

    // torch::Tensor d_loss = torch::nn::functional::cross_entropy(prediction,
    // t_labs);
    // TODO potential -- Change this to long
    auto criterion = torch::nn::CrossEntropyLoss();
    // torch::Tensor d_loss = torch::nll_loss(prediction_train, labels_train);
    torch::Tensor d_loss = criterion(prediction_train, labels_train);

    // d_loss.backward();

    // optimizer.step();

    cudaDeviceSynchronize();
    end_train = get_time();

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

    // Print the results of the precompute function
  }

  CUDA_CHECK(cudaFree(dA_csrOffsets));
  CUDA_CHECK(cudaFree(dA_values));
  CUDA_CHECK(cudaFree(dA_columns));
  CUDA_CHECK(cudaFree(dB));

  std::cout << "Inference: " << calc_mean(times_arr) << ","
            << calc_std(times_arr) << std::endl;
  std::cout << "Train: " << calc_mean(times_arr_train) << ","
            << calc_std(times_arr_train) << std::endl;
}