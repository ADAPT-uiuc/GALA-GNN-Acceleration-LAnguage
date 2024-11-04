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

torch::Tensor edge_forward_gcn(torch::Tensor input_dense1,
                               torch::Tensor input_dense2,
                               torch::Tensor offset_graph,
                               torch::Tensor columns_graph,
                               torch::Tensor value_graph, torch::Tensor bounds,
                               int nrows, int segments, bool is_directed) {
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

  // cudaStream_t stream1;
  // cudaStreamCreate(&stream1);
  // dim3 gridDim(((int)(nrows - 1) / 8) + 1);
  // dim3 blockDim(32, 8);
  // default_function_kernel_sddvv_plus<<<gridDim, blockDim, 0, stream1>>>(
  //     oden_array, offset_ptr, iden_ptr1, iden_ptr2, col_ptr, nrows);

  for (int i = 0; i < segments; i++) {
    int i1 = i;
    int start_vals = bounds_ptr[i1 * 2];
    int end_vals = bounds_ptr[i1 * 2 + 1];
    int nvals = end_vals - start_vals;

    cudaStream_t stream1, stream2, stream3;

    if (is_directed) {
      cudaStreamCreate(&stream1);
      dim3 gridDim(((int)(nrows - 1) / 8) + 1);
      dim3 blockDim(32, 8);
      default_function_kernel_sddvv_mult_undir<<<gridDim, blockDim, 0,
                                                 stream1>>>(
          &oden_array[start_vals], &offset_ptr[i1 * (nrows + 1)], iden_ptr1,
          iden_ptr2, &col_ptr[start_vals], nrows);
    } else {
      cudaStreamCreate(&stream1);
      dim3 gridDim(((int)(nrows - 1) / 8) + 1);
      dim3 blockDim(32, 8);
      default_function_kernel_sddvv_mult_undir<<<gridDim, blockDim, 0,
                                                 stream1>>>(
          &oden_array[start_vals], &offset_ptr[i1 * (nrows + 1)], iden_ptr1,
          iden_ptr2, &col_ptr[start_vals], nrows);
    }
  }

  return output_sparse;
}

torch::Tensor gather_forward_gcn(torch::Tensor input_dense,
                                 torch::Tensor offset_graph,
                                 torch::Tensor columns_graph,
                                 torch::Tensor value_graph,
                                 torch::Tensor bounds, int nrows, int segments,
                                 bool is_directed) {
  auto full_iden = input_dense.numel();

  // int nrows = global_nrows;
  // int segments = global_segments;
  // bool is_directed = global_is_directed;

  // torch::Tensor offset_graph = global_offset_graph;
  // torch::Tensor columns_graph = global_columns_graph;
  // torch::Tensor value_graph = global_value_graph;
  // torch::Tensor bounds = global_bounds;

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
    int end_vals = bounds_ptr[i1 * 2 + 1];
    int nvals = end_vals - start_vals;

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

  return {output_dense};
}

class GatherForward : public torch::autograd::Function<GatherForward> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor input_dense, int li) {
    ctx->saved_data["li"] = li;
    torch::Tensor offset_graph = global_offset_graph[2 * li];
    torch::Tensor columns_graph = global_columns_graph[2 * li];
    torch::Tensor value_graph = global_value_graph[2 * li];
    torch::Tensor bounds = global_bounds[2 * li];
    int segments = global_segments[2 * li];
    return gather_forward_gcn(input_dense, offset_graph, columns_graph,
                              value_graph, bounds, global_nrows, segments,
                              global_is_directed);
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    torch::Tensor input_dense = grad_outputs[0];
    int li = ctx->saved_data["li"].toInt();
    torch::Tensor offset_graph = global_offset_graph[2 * li + 1];
    torch::Tensor columns_graph = global_columns_graph[2 * li + 1];
    torch::Tensor value_graph = global_value_graph[2 * li + 1];
    torch::Tensor bounds = global_bounds[2 * li + 1];
    int segments = global_segments[2 * li + 1];
    return {gather_forward_gcn(input_dense, offset_graph, columns_graph,
                               value_graph, bounds, global_nrows, segments,
                               global_is_directed),
            torch::Tensor()};
  }
};

struct GCN : torch::nn::Module {
  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  int in_feat_size, hidden_feat_size, out_feat_size;
  bool directed;

  GCN(int in_size, int hidden_size, int out_size, bool dir) {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(in_size, hidden_size));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_size, out_size));
    in_feat_size = in_size;
    hidden_feat_size = hidden_size;
    out_feat_size = out_size;
    directed = dir;
    global_is_directed = dir;
  }

  std::vector<torch::Tensor> forward(torch::Tensor input_dense, // B
                                     torch::Tensor degree, int ep, int mod_v) {

    if (ep % mod_v == 0) {
      torch::Tensor res = fc1->forward(input_dense);
      res = torch::relu(res);
      if (hidden_feat_size > out_feat_size){
        res = fc2->forward(res);
        res = GatherForward::apply(res, 0);
      } else {
        res = GatherForward::apply(res, 0);
        res = fc2->forward(res);
      }
      return {torch::log_softmax(res, /*dim=*/1)};
    } else {
      torch::Tensor res = fc1->forward(input_dense);
      res = torch::relu(res);
      if (hidden_feat_size > out_feat_size){
        res = fc2->forward(res);
        res = GatherForward::apply(res, 2);
      } else {
        res = GatherForward::apply(res, 2);
        res = fc2->forward(res);

      }
      return {torch::log_softmax(res, /*dim=*/1)};
    }
  }
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

  // Column tiling
  iT cols_per_tile_b = stoi(string(argv[4]));

  // Const settings
  int skip_cache_warmup = 5;

  // Set the graph is directed
  bool directed = false;

  std::string filename;
  SM adj;
  filename = path;
  readSM_npy32<SM>(filename, &adj);

  adj.set_all(1);

  // Adj info
  iT nrows = adj.nrows();
  iT ncols = adj.ncols();
  nT nvals = adj.nvals();

  // Load input
  DM input_emb;
  readDM_npy<DM>(filename + "Feat.npy", &input_emb,
                 DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
  iT emb_size = input_emb.ncols();
  DL labels;
  readDM_npy<DL>(filename + "Lab.npy", &labels, DL::DENSE_MTX_TYPE::RM);

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

  // -------------- Get the graph subsets from mask -----------
  double start_pre, end_pre;
  start_pre = get_time();
  std::vector<SM *> forward_adj;
  std::vector<SM *> backward_adj;
  getMaskSubgraphs(&adj, &train_mask, 2, forward_adj, backward_adj);
  SM *adj_1f = forward_adj[1];
  SM *adj_1b = backward_adj[1];
  SM *adj_2f = forward_adj[0];
  SM *adj_2b = backward_adj[0];
  nT nvals0 = adj_1f->nvals();
  nT nvals1 = adj_2f->nvals();
  end_pre = get_time();
  std::cout << "Preprocess time: " << end_pre - start_pre << std::endl;
  //------------------------------------------------------------
  // -------------- Tiling the input graph --------------------
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).requires_grad(false);
  auto options_float =
      torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);
  ///--------------------------
  torch::Tensor total_offsets;
  torch::Tensor total_cols;
  torch::Tensor total_vals;
  torch::Tensor total_bounds;
  // Can have individual tiles here
  std::vector<iT> tile_offsets =
      static_ord_col_breakpoints<SM>(&adj, cols_per_tile);
  iT segments = tile_offsets.size() - 1;
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
  // 1f -----
  torch::Tensor total_offsets_1f;
  torch::Tensor total_cols_1f;
  torch::Tensor total_vals_1f;
  torch::Tensor total_bounds_1f;
  // Can have individual tiles here
  std::vector<iT> tile_offsets_1f =
      static_ord_col_breakpoints<SM>(adj_1f, cols_per_tile);
  iT segments_1f = tile_offsets_1f.size() - 1;
  // The first and last value of this should also give the offsets for the
  // columns and vals
  total_offsets_1f = torch::zeros({(nrows + 1) * (segments_1f)}, options_int);
  total_cols_1f = torch::zeros({nvals0}, options_int);
  total_vals_1f = torch::zeros({nvals0}, options_float);
  total_bounds_1f = torch::zeros({2 * (segments_1f)}, options_int);
  ord_col_tiling_torch(tile_offsets_1f, total_offsets_1f, total_cols_1f,
                       total_vals_1f, total_bounds_1f, adj_1f);
  iT *offset_ptr_1f = total_offsets_1f.data_ptr<iT>();
  iT *col_ptr_1f = total_cols_1f.data_ptr<iT>();
  vT *val_ptr_1f = total_vals_1f.data_ptr<vT>();
  // 1b -----
  torch::Tensor total_offsets_1b;
  torch::Tensor total_cols_1b;
  torch::Tensor total_vals_1b;
  torch::Tensor total_bounds_1b;
  // Can have individual tiles here
  std::vector<iT> tile_offsets_1b =
      static_ord_col_breakpoints<SM>(adj_1b, cols_per_tile_b);
  iT segments_1b = tile_offsets_1b.size() - 1;
  // The first and last value of this should also give the offsets for the
  // columns and vals
  total_offsets_1b = torch::zeros({(nrows + 1) * (segments_1b)}, options_int);
  total_cols_1b = torch::zeros({nvals0}, options_int);
  total_vals_1b = torch::zeros({nvals0}, options_float);
  total_bounds_1b = torch::zeros({2 * (segments_1b)}, options_int);
  ord_col_tiling_torch(tile_offsets_1b, total_offsets_1b, total_cols_1b,
                       total_vals_1b, total_bounds_1b, adj_1b);
  iT *offset_ptr_1b = total_offsets_1b.data_ptr<iT>();
  iT *col_ptr_1b = total_cols_1b.data_ptr<iT>();
  vT *val_ptr_1b = total_vals_1b.data_ptr<vT>();
  // 2f -----
  torch::Tensor total_offsets_2f;
  torch::Tensor total_cols_2f;
  torch::Tensor total_vals_2f;
  torch::Tensor total_bounds_2f;
  // Can have individual tiles here
  std::vector<iT> tile_offsets_2f =
      static_ord_col_breakpoints<SM>(adj_2f, cols_per_tile);
  iT segments_2f = tile_offsets_2f.size() - 1;
  // The first and last value of this should also give the offsets for the
  // columns and vals
  total_offsets_2f = torch::zeros({(nrows + 1) * (segments_2f)}, options_int);
  total_cols_2f = torch::zeros({nvals1}, options_int);
  total_vals_2f = torch::zeros({nvals1}, options_float);
  total_bounds_2f = torch::zeros({2 * (segments_2f)}, options_int);
  ord_col_tiling_torch(tile_offsets_2f, total_offsets_2f, total_cols_2f,
                       total_vals_2f, total_bounds_2f, adj_2f);
  iT *offset_ptr_2f = total_offsets_2f.data_ptr<iT>();
  iT *col_ptr_2f = total_cols_2f.data_ptr<iT>();
  vT *val_ptr_2f = total_vals_2f.data_ptr<vT>();
  // 1b -----
  torch::Tensor total_offsets_2b;
  torch::Tensor total_cols_2b;
  torch::Tensor total_vals_2b;
  torch::Tensor total_bounds_2b;
  // Can have individual tiles here
  std::vector<iT> tile_offsets_2b =
      static_ord_col_breakpoints<SM>(adj_2b, cols_per_tile_b);
  iT segments_2b = tile_offsets_2b.size() - 1;
  // The first and last value of this should also give the offsets for the
  // columns and vals
  total_offsets_2b = torch::zeros({(nrows + 1) * (segments_2b)}, options_int);
  total_cols_2b = torch::zeros({nvals1}, options_int);
  total_vals_2b = torch::zeros({nvals1}, options_float);
  total_bounds_2b = torch::zeros({2 * (segments_2b)}, options_int);
  ord_col_tiling_torch(tile_offsets_2b, total_offsets_2b, total_cols_2b,
                       total_vals_2b, total_bounds_2b, adj_2b);
  iT *offset_ptr_2b = total_offsets_2b.data_ptr<iT>();
  iT *col_ptr_2b = total_cols_2b.data_ptr<iT>();
  vT *val_ptr_2b = total_vals_2b.data_ptr<vT>();
  global_bounds.push_back(total_bounds);
  global_bounds.push_back(total_bounds);
  global_bounds.push_back(total_bounds_1f);
  global_bounds.push_back(total_bounds_1b);
  global_bounds.push_back(total_bounds_2f);
  global_bounds.push_back(total_bounds_2b);

  std::cout << adj.nrows() << " " << adj.ncols() << " " << adj.nvals()
            << std::endl;
  int classes =
      *std::max_element(labels.vals_ptr(), labels.vals_ptr() + labels.nvals()) +
      1;
  std::cout << "Classes: " << classes << std::endl;
  torch::Device device(torch::kCUDA);

  iT *dA_csrOffsets, *dA_columns, *dA_csrOffsets_1f, *dA_columns_1f,
      *dA_csrOffsets_1b, *dA_columns_1b, *dA_csrOffsets_2f, *dA_columns_2f,
      *dA_csrOffsets_2b, *dA_columns_2b, *dL;
  vT *dA_values, *dA_values_1f, *dA_values_1b, *dA_values_2f, *dA_values_2b,
      *dB;
  bool *d_train_mask, *d_valid_mask, *d_test_mask;

  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets,
                        ((nrows + 1) * segments) * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_columns, nvals * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_values, nvals * sizeof(vT)));
  // 1F
  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets_1f, ((nrows + 1) * segments_1f) * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_columns_1f, nvals0 * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_values_1f, nvals0 * sizeof(float)));
  // 1B
  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets_1b, ((nrows + 1) * segments_1b) * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_columns_1b, nvals0 * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_values_1b, nvals0 * sizeof(float)));
  // 2F
  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets_2f, ((nrows + 1) * segments_2f) * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_columns_2f, nvals1 * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_values_2f, nvals1 * sizeof(float)));
  // 1B
  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets_2b, ((nrows + 1) * segments_2b) * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_columns_2b, nvals1 * sizeof(iT)));
  CUDA_CHECK(cudaMalloc((void **)&dA_values_2b, nvals1 * sizeof(float)));
  //--------
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
  // 1F
  CUDA_CHECK(cudaMemcpy(dA_csrOffsets_1f, offset_ptr_1f,
                        ((nrows + 1) * segments_1f) * sizeof(iT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_columns_1f, col_ptr_1f, nvals0 * sizeof(iT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_values_1f, val_ptr_1f, nvals0 * sizeof(float),
                        cudaMemcpyHostToDevice));
  // 1B
  CUDA_CHECK(cudaMemcpy(dA_csrOffsets_1b, offset_ptr_1b,
                        ((nrows + 1) * segments_1b) * sizeof(iT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_columns_1b, col_ptr_1b, nvals0 * sizeof(iT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_values_1b, val_ptr_1b, nvals0 * sizeof(float),
                        cudaMemcpyHostToDevice));
  // 2F
  CUDA_CHECK(cudaMemcpy(dA_csrOffsets_2f, offset_ptr_2f,
                        ((nrows + 1) * segments_2f) * sizeof(iT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_columns_2f, col_ptr_2f, nvals1 * sizeof(iT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_values_2f, val_ptr_2f, nvals1 * sizeof(float),
                        cudaMemcpyHostToDevice));
  // 2B
  CUDA_CHECK(cudaMemcpy(dA_csrOffsets_2b, offset_ptr_2b,
                        ((nrows + 1) * segments_2b) * sizeof(iT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_columns_2b, col_ptr_2b, nvals1 * sizeof(iT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_values_2b, val_ptr_2b, nvals1 * sizeof(float),
                        cudaMemcpyHostToDevice));
  // --------------------
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
  // 1f
  torch::Tensor t_offsets_1f = torch::from_blob(
      dA_csrOffsets_1f, {(nrows + 1) * segments_1f}, options_cu_int);
  torch::Tensor t_cols_1f =
      torch::from_blob(dA_columns_1f, {nvals0}, options_cu_int);
  // 1b
  torch::Tensor t_offsets_1b = torch::from_blob(
      dA_csrOffsets_1b, {(nrows + 1) * segments_1b}, options_cu_int);
  torch::Tensor t_cols_1b =
      torch::from_blob(dA_columns_1b, {nvals0}, options_cu_int);
  // 2f
  torch::Tensor t_offsets_2f = torch::from_blob(
      dA_csrOffsets_2f, {(nrows + 1) * segments_2f}, options_cu_int);
  torch::Tensor t_cols_2f =
      torch::from_blob(dA_columns_2f, {nvals1}, options_cu_int);
  // 2b
  torch::Tensor t_offsets_2b = torch::from_blob(
      dA_csrOffsets_2b, {(nrows + 1) * segments_2b}, options_cu_int);
  torch::Tensor t_cols_2b =
      torch::from_blob(dA_columns_2b, {nvals1}, options_cu_int);
  // ------------------
  global_offset_graph.push_back(t_offsets);
  global_offset_graph.push_back(t_offsets);
  global_offset_graph.push_back(t_offsets_1f);
  global_offset_graph.push_back(t_offsets_1b);
  global_offset_graph.push_back(t_offsets_2f);
  global_offset_graph.push_back(t_offsets_2b);
  global_columns_graph.push_back(t_cols);
  global_columns_graph.push_back(t_cols);
  global_columns_graph.push_back(t_cols_1f);
  global_columns_graph.push_back(t_cols_1b);
  global_columns_graph.push_back(t_cols_2f);
  global_columns_graph.push_back(t_cols_2b);

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
  // 1f
  torch::Tensor t_vals_1f =
      torch::from_blob(dA_values_1f, {nvals0}, options_cu_float_ngrad);
  torch::Tensor t_vals_1b =
      torch::from_blob(dA_values_1b, {nvals0}, options_cu_float_ngrad);
  torch::Tensor t_vals_2f =
      torch::from_blob(dA_values_2f, {nvals1}, options_cu_float_ngrad);
  torch::Tensor t_vals_2b =
      torch::from_blob(dA_values_2b, {nvals1}, options_cu_float_ngrad);
  global_value_graph.push_back(t_vals);
  global_value_graph.push_back(t_vals);
  global_value_graph.push_back(t_vals_1f);
  global_value_graph.push_back(t_vals_1b);
  global_value_graph.push_back(t_vals_2f);
  global_value_graph.push_back(t_vals_2b);
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
  // TICM
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat)
                     .requires_grad(false)
                     .device(torch::kCUDA, 0);
  global_nrows = nrows;
  global_segments.push_back(segments);
  global_segments.push_back(segments);
  global_segments.push_back(segments_1f);
  global_segments.push_back(segments_1b);
  global_segments.push_back(segments_2f);
  global_segments.push_back(segments_2b);

  directed = true;
  // global_bounds = total_bounds;
  // global_offset_graph = t_offsets;
  // global_value_graph = t_vals;
  // global_columns_graph = t_cols;
  auto t_ones = torch::ones({nrows, 1}, options);
  torch::Tensor t_degree =
      gather_forward_gcn(t_ones, t_offsets, t_cols, t_vals, total_bounds, nrows,
                         segments, directed);
  t_degree = torch::pow(t_degree, -0.5).detach();
  torch::Tensor t_sp_vals =
      edge_forward_gcn(t_degree, t_degree, t_offsets, t_cols, t_vals,
                       total_bounds, nrows, segments, directed);
  global_value_graph[0] = t_sp_vals;
  global_value_graph[1] = t_sp_vals;
  global_value_graph[2] = edge_forward_gcn(t_degree, t_degree, t_offsets_1f, t_cols_1f, global_value_graph[2],
                       total_bounds_1f, nrows, segments_1f, directed);
  global_value_graph[3] = edge_forward_gcn(t_degree, t_degree, t_offsets_1b, t_cols_1b, global_value_graph[3],
                       total_bounds_1b, nrows, segments_1b, directed);
  global_value_graph[4] = edge_forward_gcn(t_degree, t_degree, t_offsets_2f, t_cols_2f, global_value_graph[4],
                       total_bounds_2f, nrows, segments_2f, directed);
  global_value_graph[5] = edge_forward_gcn(t_degree, t_degree, t_offsets_2b, t_cols_2b, global_value_graph[5],
                       total_bounds_2b, nrows, segments_2b, directed);
  torch::Tensor norm_input =
      gather_forward_gcn(t_iden, t_offsets, t_cols, t_sp_vals,
                         total_bounds, nrows, segments, directed);
  auto net = std::make_shared<GCN>(emb_size, 32, classes, directed);
  cudaDeviceSynchronize();
  end_init = get_time();

  std::cout << "Initialization time: " << end_init - start_init << std::endl;

  net->to(device);

  std::cout << "Initial Memory Usage: " << std::endl;
  printMemoryUsage();


  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::Adam optimizer(
      net->parameters(), torch::optim::AdamOptions(1e-2).weight_decay(5e-4));

  double start, end;
  double start_train, end_train;
  val_t randVal;
  std::vector<double> times_arr, times_arr_train;

  int mod_v = 5;

  for (size_t epoch = 1; epoch <= num_iters; ++epoch) {
    // Reset gradients.
    optimizer.zero_grad();
    // Execute the model on the input data.
    cudaDeviceSynchronize();
    start = get_time();
    torch::Tensor prediction =
        net->forward(norm_input, t_degree, epoch, mod_v)[0];

    cudaDeviceSynchronize();
    end = get_time();

    std::cout << "After forward: " << std::endl;
    printMemoryUsage();


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

    std::cout << "After backward: " << std::endl;
    printMemoryUsage();


    if (epoch % mod_v == 0) {
      torch::Tensor prediction_test = prediction.index({t_test_mask});
      torch::Tensor labels_test = t_labs.index({t_test_mask});

      auto [pred_val, pred_idx] = torch::max({prediction_test}, 1);

      auto correct = torch::sum(pred_idx == labels_test);

      std::cout << "Epoch " << epoch << " Loss: " << d_loss.item<val_t>()
                << " Accuracy: "
                << (correct.item<val_t>() * 100.0 / labels_test.sizes()[0])
                << std::endl;
    } else {
      std::cout << "Epoch " << epoch << " Loss: " << d_loss.item<val_t>()
                << std::endl;
    }

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
  std::cout << "Total: " << calc_mean(times_arr) + calc_mean(times_arr_train) << std::endl;
}