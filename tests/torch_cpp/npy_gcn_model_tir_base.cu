//
// Created by damitha on 5/12/24.
//
// Define a new Module.
#include "kernels.cu"
#include <algorithm>

typedef int ind1_t;
typedef int ind2_t;
typedef float val_t;

// Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DM;
typedef DenseMatrix<ind1_t, ind2_t, ind1_t> DL;
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM;

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

  //	dim3 gridDim(((int)nrows + 1) / 8, ((int)dcols + 1) / 64);
  //    dim3 blockDim(32, 8);
  //    default_function_kernel_if<<<gridDim, blockDim>>>(oden_array,
  //                                                      offset_ptr,
  //                                                      val_ptr,
  //                                                      iden_ptr,
  //                                                      col_ptr,
  //                                                      nrows,
  //                                                      dcols);

  cudaStream_t stream1, stream2, stream3;

  if ((int)dcols / 64) {
    cudaStreamCreate(&stream1);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 64);
    dim3 blockDim(32, 8);
    default_function_kernel64<<<gridDim, blockDim, 0, stream1>>>(
        oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols);

    if ((dcols % 64) > 32) {
      cudaStreamCreate(&stream2);
      dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
      dim3 blockDim_rem(32, 8);
      default_function_kernel32_offset<<<gridDim_rem, blockDim_rem, 0,
                                         stream2>>>(
          oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols,
          ((int)dcols / 64) * 64);
      if ((dcols % 32) > 0) {
        cudaStreamCreate(&stream3);
        dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
        dim3 blockDim_rem(dcols % 32, 8);
        default_function_kernel_rem<<<gridDim_rem, blockDim_rem, 0, stream3>>>(
            oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols,
            (((int)dcols / 64) * 64) + 32);
      }
    } else if ((dcols % 64) > 0) {
      cudaStreamCreate(&stream2);
      dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
      dim3 blockDim_rem(dcols % 64, 8);
      default_function_kernel_rem<<<gridDim_rem, blockDim_rem, 0, stream2>>>(
          oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols,
          ((int)dcols / 64) * 64);
    }
  } else {
    if ((int)dcols / 32) {
      cudaStreamCreate(&stream1);
      dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, (int)dcols / 32);
      dim3 blockDim_rem(32, 8);
      default_function_kernel32<<<gridDim_rem, blockDim_rem, 0, stream1>>>(
          oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols);
      if ((dcols % 32) > 0) {
        cudaStreamCreate(&stream2);
        dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
        dim3 blockDim_rem(dcols % 32, 8);
        default_function_kernel_rem<<<gridDim_rem, blockDim_rem, 0, stream2>>>(
            oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols,
            (((int)dcols / 32) * 32));
      }
    } else {
      cudaStreamCreate(&stream1);
      dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
      dim3 blockDim_rem(dcols % 32, 8);
      default_function_kernel_rem<<<gridDim_rem, blockDim_rem, 0, stream1>>>(
          oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols, 0);
    }
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

    torch::Tensor msg_aggr = gather_forward_gcn(norm_input, offset_graph,
                                                columns_graph, value_graph)[0];

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

    msg_aggr = gather_forward_gcn(norm_input, offset_graph, columns_graph,
                                  value_graph)[0];

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
  // Timing configs
  int num_iters = stoi(string(argv[2]));

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
                 DenseMatrix<ind1_t, ind2_t, ind1_t>::DENSE_MTX_TYPE::RM);

  std::cout << adj.nrows() << " " << adj.ncols() << " " << adj.nvals()
            << std::endl;
  int classes =
      *std::max_element(labels.vals_ptr(), labels.vals_ptr() + labels.nvals()) +
      1;
  std::cout << "Classes: " << classes << std::endl;
  auto wsum_aggr = wsumAgg<val_t, val_t, ind2_t>;

  torch::Device device(torch::kCUDA);
  // Create a new Net.

  double start_init, end_init;
  cudaDeviceSynchronize();
  start_init = get_time();
  auto net = std::make_shared<GCN>(emb_size, 32, classes, false);
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
      torch::from_blob(dB, {nrows, emb_size}, options_cu_float);

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

    // randVal = prediction[nrows - 1][emb_size - 1].item<val_t>();

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

  std::cout << calc_mean(times_arr) << "," << calc_std(times_arr) << std::endl;
}