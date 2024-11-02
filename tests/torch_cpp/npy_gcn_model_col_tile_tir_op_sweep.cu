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
int global_segments;
bool global_is_directed;

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
      if (((int)dcols / 1024) && ((int)dcols % 1024 < 32)){
        cudaStreamCreate(&stream1);
        dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 1024);
        dim3 blockDim(32, 8);
        default_function_kernel64_undir<<<gridDim, blockDim, 0, stream1>>>(
            oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
            &col_ptr[start_vals], nrows, dcols);
        if ((dcols % 32) > 0) {
            cudaStreamCreate(&stream3);
            dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
            dim3 blockDim_rem(dcols % 32, 8);
            default_function_kernel_rem_undir<<<gridDim_rem, blockDim_rem, 0,
                                                stream3>>>(
                oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
                &col_ptr[start_vals], nrows, dcols,
                (((int)dcols / 1024) * 1024));
        }

      }
      else if (((int)dcols / 576) && ((int)dcols % 576 < 32)){
        cudaStreamCreate(&stream1);
        dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 576);
        dim3 blockDim(32, 8);
        default_function_kernel64_undir<<<gridDim, blockDim, 0, stream1>>>(
            oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
            &col_ptr[start_vals], nrows, dcols);
        if ((dcols % 32) > 0) {
            cudaStreamCreate(&stream3);
            dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
            dim3 blockDim_rem(dcols % 32, 8);
            default_function_kernel_rem_undir<<<gridDim_rem, blockDim_rem, 0,
                                                stream3>>>(
                oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
                &col_ptr[start_vals], nrows, dcols,
                (((int)dcols / 576) * 576));
        }

      }
      else if (((int)dcols / 512) && ((int)dcols % 512 < 32)){
        cudaStreamCreate(&stream1);
        dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 512);
        dim3 blockDim(32, 8);
        default_function_kernel64_undir<<<gridDim, blockDim, 0, stream1>>>(
            oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
            &col_ptr[start_vals], nrows, dcols);
        if ((dcols % 32) > 0) {
            cudaStreamCreate(&stream3);
            dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);
            dim3 blockDim_rem(dcols % 32, 8);
            default_function_kernel_rem_undir<<<gridDim_rem, blockDim_rem, 0,
                                                stream3>>>(
                oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
                &col_ptr[start_vals], nrows, dcols,
                (((int)dcols / 512) * 512));
        }

      }
      else if (((int)dcols / 256) && ((int)dcols % 256 == 0)){
        cudaStreamCreate(&stream1);
        dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 256);
        dim3 blockDim(32, 8);
        default_function_kernel64_undir<<<gridDim, blockDim, 0, stream1>>>(
            oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
            &col_ptr[start_vals], nrows, dcols);

      }
      else if (((int)dcols / 128) && ((int)dcols % 128 == 0)){
        cudaStreamCreate(&stream1);
        dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 128);
        dim3 blockDim(32, 8);
        default_function_kernel64_undir<<<gridDim, blockDim, 0, stream1>>>(
            oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr,
            &col_ptr[start_vals], nrows, dcols);

      }
      else if ((int)dcols / 64) {
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

  return {output_dense};
}

class GatherForward : public torch::autograd::Function<GatherForward> {
public:
  static torch::Tensor
  forward(torch::autograd::AutogradContext *ctx, torch::Tensor input_dense,
          torch::Tensor offset_graph, torch::Tensor columns_graph,
          torch::Tensor value_graph, torch::Tensor bounds) {
    ctx->save_for_backward({offset_graph, columns_graph, value_graph, bounds});
    return gather_forward_gcn(input_dense, offset_graph, columns_graph,
                              value_graph, bounds, global_nrows,
                              global_segments, global_is_directed);
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    torch::Tensor input_dense = grad_outputs[0];
    auto saved = ctx->get_saved_variables();
    torch::Tensor offset_graph = saved[0];
    torch::Tensor columns_graph = saved[1];
    torch::Tensor value_graph = saved[2];
    torch::Tensor bounds = saved[3];
    // std::cout << "grad: " << input_dense.sizes()[0] << "," <<
    // input_dense.sizes()[1] << std::endl;

    // std::cout << "grad: " << input_dense[0][0].item<val_t>() << "," <<
    // input_dense[0][1].item<val_t>() << "," << input_dense[1][0].item<val_t>()
    // << std::endl;
    return {gather_forward_gcn(input_dense, offset_graph, columns_graph,
                               value_graph, bounds, global_nrows,
                               global_segments, global_is_directed),
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

struct GCN : torch::nn::Module {
  GCN(int in_size, int hidden_size, int out_size, bool dir, int nl) {
    // Construct and register two Linear submodules.
    if (nl == 0){
      fc_vec.push_back(register_module("fc0", torch::nn::Linear(in_size, out_size)));
    } else {
      fc_vec.push_back(register_module("fc0", torch::nn::Linear(in_size, hidden_size)));
      for (int i = 0; i < nl - 1; i++){
        std::string m_name =  "fc" + std::to_string(i + 1);
        fc_vec.push_back(register_module(m_name, torch::nn::Linear(hidden_size, hidden_size)));
      }
      std::string m_name_last =  "fc" + std::to_string(nl);
      fc_vec.push_back(register_module(m_name_last, torch::nn::Linear(hidden_size, out_size)));
    }
    fx0 = register_module("fx0", torch::nn::Linear(in_size, out_size));
    in_feat_size = in_size;
    hidden_feat_size = hidden_size;
    out_feat_size = out_size;
    directed = dir;
    global_is_directed = dir;
    n_layers = nl;
  }

  std::vector<torch::Tensor>
  forward(torch::Tensor input_dense,   // B
          torch::Tensor offset_graph,  // A_sparse_offset
          torch::Tensor columns_graph, // A_sparse_col_ids
          torch::Tensor value_graph,   // A_sparse_values
          torch::Tensor bounds,        // A_sparse_tile_bounds
          int nrows, int segments) {

    global_nrows = nrows;
    global_segments = segments;

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat)
                       .requires_grad(false)
                       .device(torch::kCUDA, 0);
    auto ones = torch::ones({nrows, 1}, options);
    torch::Tensor degree =
        gather_forward_gcn(ones, offset_graph, columns_graph, value_graph,
                           bounds, nrows, segments, directed);

    degree = torch::pow(degree, -0.5);

    torch::Tensor res;

    // res = fc_vec[0]->forward(res);
    // res = degree * res;
    // res = GatherForward::apply(res, offset_graph, columns_graph, value_graph,
    //                           bounds);
    // res = degree * res;
    // return {torch::log_softmax(res, /*dim=*/1)};    
    // std::cout << "run this"_ << n_layers << std::endl;
    if (n_layers == 0) {
      // std::cout << "run this" << std::endl;
      if (in_feat_size > out_feat_size){
        res = fc_vec[0]->forward(input_dense);
        // std::cout << "run this" << std::endl;
        res = degree * res;
        // std::cout << "run this" << std::endl;
        res = GatherForward::apply(res, offset_graph, columns_graph, value_graph,
                                  bounds);
        // std::cout << "run this" << std::endl;
        res = degree * res;
        // std::cout << "run this" << std::endl;
        return {torch::log_softmax(res, /*dim=*/1)};    
      } else {
        res = degree * input_dense;
        res = GatherForward::apply(res, offset_graph, columns_graph, value_graph,
                                  bounds);
        res = degree * res;
        res = fc_vec[0]->forward(res); 
        return {torch::log_softmax(res, /*dim=*/1)};    
      }
    } else {
      torch::Tensor res;
      if (in_feat_size > hidden_feat_size){
        res = fc_vec[0]->forward(input_dense);
        res = degree * res;
        res = GatherForward::apply(res, offset_graph, columns_graph, value_graph,
                                  bounds);
        res = degree * res;
        res = torch::relu(res);
      } else {
        res = degree * input_dense;

        res = GatherForward::apply(res, offset_graph, columns_graph, value_graph,
                                  bounds);
        res = degree * res;
        res = fc_vec[0]->forward(res);
        res = torch::relu(res); 
      }
      for (int i = 0; i < n_layers - 1; i++){
        res = degree * res;
        res = GatherForward::apply(res, offset_graph, columns_graph, value_graph,
                                  bounds);
        res = degree * res;
        res = fc_vec[i + 1]->forward(res);
        res = torch::relu(res);
      }
      if (hidden_feat_size > out_feat_size){
        res = fc_vec[n_layers]->forward(res);
        res = degree * res;
        res = GatherForward::apply(res, offset_graph, columns_graph, value_graph,
                                  bounds);
        res = degree * res;
        return {torch::log_softmax(res, /*dim=*/1)};    
      } else {
        res = degree * res;
        res = GatherForward::apply(res, offset_graph, columns_graph, value_graph,
                                  bounds);
        res = degree * res;
        res = fc_vec[n_layers]->forward(res); 
        return {torch::log_softmax(res, /*dim=*/1)};    
      }
    }
  }

  // Use one of many "standard library" modules.
  std::vector<torch::nn::Linear> fc_vec;
  torch::nn::Linear fx0{nullptr};
  int in_feat_size, hidden_feat_size, out_feat_size, n_layers;
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

  int n_hidden = stoi(string(argv[4]));
  int n_layers = stoi(string(argv[5]));
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

  // std::cout << train_mask.vals_ptr()[0] << "," << train_mask.vals_ptr()[1]
  //           << "," << train_mask.vals_ptr()[2] << ","
  //           << train_mask.vals_ptr()[3] << std::endl;

  // if (do_reorder) {
  //   std::unique_ptr<vint[]> perm_rabbit;
  //   auto nvals_var = adj.nvals();
  //   SM::itype *col_ids_var = adj.ids_ptr();
  //   auto vals_var = adj.vals_ptr();
  //   SM_t::itype *row_ids_var;
  //   get_row_ids<SM>(&adj, row_ids_var);
  //   get_perm_graph<SM>(nvals_var, row_ids_var, col_ids_var, vals_var,
  //                        perm_rabbit);
  //   SM::itype perm[adj.nrows()];
  //   for (SM::ntype p_i = 0; p_i < adj.nrows(); p_i++) {
  //     perm[p_i] = (SM::itype)perm_rabbit[p_i];
  //   }
  //   rowReorderToTorch(&adj, &input_emb, &train_mask, &valid_mask, &test_mask,
  //                &labels, perm);
  // }

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
  auto net = std::make_shared<GCN>(emb_size, n_hidden, classes, false, n_layers);
  cudaDeviceSynchronize();
  end_init = get_time();

  std::cout << "Initialization time: " << end_init - start_init << std::endl;

  net->to(device);

  // std::cout << "Initial Memory Usage: " << std::endl;
  // printMemoryUsage();

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::Adam optimizer(
      net->parameters(), torch::optim::AdamOptions(1e-2).weight_decay(5e-4));

  double start, end;
  double start_train, end_train;
  std::vector<double> times_arr, times_arr_train;
  for (size_t epoch = 1; epoch <= num_iters; ++epoch) {
    // Reset gradients.
    optimizer.zero_grad();
    // Execute the model on the input data.
    cudaDeviceSynchronize();
    start = get_time();
    torch::Tensor prediction = net->forward(t_iden, t_offsets, t_cols, t_vals,
                                            total_bounds, nrows, segments)[0];

    cudaDeviceSynchronize();
    end = get_time();

    // std::cout << "After forward: " << std::endl;
    // printMemoryUsage();

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

    // std::cout << "After backward: " << std::endl;
    // printMemoryUsage();

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