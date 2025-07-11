#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>
#include <torch/script.h>
#include <cmath>
#include <iostream>
#include <parallel/algorithm>
#include <vector>
#include <bits/stdc++.h>
#include <omp.h>
#include <stdlib.h>
#include <torch/torch.h>
#include "../src/formats/csrc_matrix.h"
#include "../src/formats/dense_matrix.h"
#include "../src/ops/aggregators.h"
#include "../src/ops/tiling.h"
#include "../src/utils/mtx_io.h"
#include "../tests/common.h"

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
int global_classes;
int global_emb_size;
int global_ra;
int global_rb;
std::vector<int> global_segments;
bool global_is_directed;

std::vector<torch::Tensor> global_offset_graph;
std::vector<torch::Tensor> global_columns_graph;
std::vector<torch::Tensor> global_value_graph;
std::vector<torch::Tensor> global_bounds;


#define CUDA_CHECK(func)\
  do {\
    cudaError_t status = (func);\
    if (status != cudaSuccess) {\
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,\
             cudaGetErrorString(status), status);\
      exit(EXIT_FAILURE);\
    }\
  } while (0)\

#define CUSPARSE_CHECK(func)\
  do {\
    cusparseStatus_t status = (func);\
    if (status != CUSPARSE_STATUS_SUCCESS) {\
      printf("CUSPARSE failed at line %d with error: %s (%d)\n", __LINE__,\
             cusparseGetErrorString(status), status);\
      exit(EXIT_FAILURE);\
    }\
  } while (0)
extern "C" __global__ void __launch_bounds__(256)
aggregate_node_mul_sum_direct_coarse2_kernel0(float *__restrict__ C,
                    int *__restrict__ J_indptr_data,
float *__restrict__ B,
                    int *__restrict__ J_indices_data, int nrows,
                    int dcols) {
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    float local0 = C[(((((((int)blockIdx.x) * 8)+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x)) + 0)];
        for (int j = 0;
             j <
             (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
             ++j) {
            local0 = local0 +(B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + 
                                                         ((int)threadIdx.y))])] * 
                      dcols) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x) + 0)]);
             }

C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
(((int)blockIdx.y) * 32)) +
((int)threadIdx.x) + 0)] = local0;
   }
}

extern "C" __global__ void __launch_bounds__(256)
aggregate_node_mul_sum_direct_coarse2_kernel1(float *__restrict__ C,
                    int *__restrict__ J_indptr_data,
float *__restrict__ B,
                    int *__restrict__ J_indices_data, int nrows,
                    int dcols) {
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    float local0 = C[(((((((int)blockIdx.x) * 8)+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 0)];
    float local1 = C[(((((((int)blockIdx.x) * 8)+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 32)];
        for (int j = 0;
             j <
             (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
             ++j) {
            local0 = local0 +(B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + 
                                                         ((int)threadIdx.y))])] * 
                      dcols) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x) + 0)]);
            local1 = local1 +(B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + 
                                                         ((int)threadIdx.y))])] * 
                      dcols) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x) + 32)]);
             }

C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
(((int)blockIdx.y) * 64)) +
((int)threadIdx.x) + 0)] = local0;

C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
(((int)blockIdx.y) * 64)) +
((int)threadIdx.x) + 32)] = local1;
   }
}

extern "C" __global__ void __launch_bounds__(256)
aggregate_node_mul_sum_direct_coarse2_kernel0_offset(float *__restrict__ C,
                    int *__restrict__ J_indptr_data,
float *__restrict__ B,
                    int *__restrict__ J_indices_data, int nrows,
                    int dcols, int offset) {
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    float local0 = C[(((((((int)blockIdx.x) * 8)+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x)) + 0) + offset];
for (int j = 0;
 j <
 (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
  J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
 ++j) {
local0 = local0 +(B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + 
                                             ((int)threadIdx.y))])] * 
          dcols) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x) + 0) + offset]);
             }

C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
(((int)blockIdx.y) * 32)) +
((int)threadIdx.x) + 0) + offset] = local0;
   }
}


extern "C" __global__ void __launch_bounds__(256)
aggregate_node_mul_sum_coarse2_kernel0(float *__restrict__ C,
                    int *__restrict__ J_indptr_data,
float *__restrict__ B,
                    int *__restrict__ J_indices_data, int nrows,
                    int dcols) {
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    float local0 = C[(((((((int)blockIdx.x) * 8)+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x)) + 0)];
        for (int j = 0;
             j <
             (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
             ++j) {
            local0 = local0 +(B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + 
                                                         ((int)threadIdx.y))])] * 
                      dcols) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x) + 0)]);
             }

C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
(((int)blockIdx.y) * 32)) +
((int)threadIdx.x) + 0)] = local0;
   }
}

extern "C" __global__ void __launch_bounds__(256)
aggregate_node_mul_sum_coarse2_kernel1(float *__restrict__ C,
                    int *__restrict__ J_indptr_data,
float *__restrict__ B,
                    int *__restrict__ J_indices_data, int nrows,
                    int dcols) {
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    float local0 = C[(((((((int)blockIdx.x) * 8)+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 0)];
    float local1 = C[(((((((int)blockIdx.x) * 8)+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 32)];
        for (int j = 0;
             j <
             (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
             ++j) {
            local0 = local0 +(B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + 
                                                         ((int)threadIdx.y))])] * 
                      dcols) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x) + 0)]);
            local1 = local1 +(B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + 
                                                         ((int)threadIdx.y))])] * 
                      dcols) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x) + 32)]);
             }

C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
(((int)blockIdx.y) * 64)) +
((int)threadIdx.x) + 0)] = local0;

C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
(((int)blockIdx.y) * 64)) +
((int)threadIdx.x) + 32)] = local1;
   }
}

extern "C" __global__ void __launch_bounds__(256)
aggregate_node_mul_sum_coarse2_kernel0_offset(float *__restrict__ C,
                    int *__restrict__ J_indptr_data,
float *__restrict__ B,
                    int *__restrict__ J_indices_data, int nrows,
                    int dcols, int offset) {
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    float local0 = C[(((((((int)blockIdx.x) * 8)+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x)) + 0) + offset];
for (int j = 0;
 j <
 (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
  J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
 ++j) {
local0 = local0 +(B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + 
                                             ((int)threadIdx.y))])] * 
          dcols) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x) + 0) + offset]);
             }

C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
(((int)blockIdx.y) * 32)) +
((int)threadIdx.x) + 0) + offset] = local0;
   }
}


torch::Tensor aggregate_node_mul_sum_direct_coarse2_call(torch::Tensor input_dense,
                   torch::Tensor offset_graph,
                   torch::Tensor columns_graph,
                   torch::Tensor value_graph
, torch::Tensor bounds,
 int segments) {
auto nrows = global_nrows;
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
int *bounds_ptr = bounds.data_ptr<int>();
for (int i = 0; i < segments; i++) {
  int i1 = i;
  int start_vals = bounds_ptr[i1 * 2];cudaStream_t stream0, stream1, stream2;
  if ((int)dcols / 64) {
    cudaStreamCreate(&stream2);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 64);
        dim3 blockDim(32, 8);
    aggregate_node_mul_sum_direct_coarse2_kernel1<<<gridDim, blockDim, 0, stream2>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols);
  if ((dcols % 64 ) > 32) {
    cudaStreamCreate(&stream1);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
    dim3 blockDim(32, 8);
    aggregate_node_mul_sum_direct_coarse2_kernel0_offset<<<gridDim, blockDim, 0, stream1>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /64) * 64);
  if ((dcols % 32 ) > 0) {
    cudaStreamCreate(&stream0);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
    dim3 blockDim(dcols %32, 8);
    aggregate_node_mul_sum_direct_coarse2_kernel0_offset<<<gridDim, blockDim, 0, stream0>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /32) * 32);
  }
  }
else {
  if ((dcols % 64 ) > 0) {
    cudaStreamCreate(&stream0);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
    dim3 blockDim(dcols %64, 8);
    aggregate_node_mul_sum_direct_coarse2_kernel0_offset<<<gridDim, blockDim, 0, stream0>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /64) * 64);
  }
}
  }
else {
  if ((int)dcols / 32) {
    cudaStreamCreate(&stream1);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 32);
        dim3 blockDim(32, 8);
    aggregate_node_mul_sum_direct_coarse2_kernel0<<<gridDim, blockDim, 0, stream1>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols);
  if ((dcols % 32 ) > 0) {
    cudaStreamCreate(&stream0);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
    dim3 blockDim(dcols %32, 8);
    aggregate_node_mul_sum_direct_coarse2_kernel0_offset<<<gridDim, blockDim, 0, stream0>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /32) * 32);
  }
  }
else {
  if ((int)dcols) {
    cudaStreamCreate(&stream0);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
        dim3 blockDim((int)dcols, 8);
    aggregate_node_mul_sum_direct_coarse2_kernel0<<<gridDim, blockDim, 0, stream0>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols);
  }
}
}
}return output_dense;
}
torch::Tensor aggregate_node_mul_sum_coarse2_call(torch::Tensor input_dense,
                   torch::Tensor offset_graph,
                   torch::Tensor columns_graph,
                   torch::Tensor value_graph
, torch::Tensor bounds,
 int segments) {
auto nrows = global_nrows;
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
int *bounds_ptr = bounds.data_ptr<int>();
for (int i = 0; i < segments; i++) {
  int i1 = i;
  int start_vals = bounds_ptr[i1 * 2];cudaStream_t stream0, stream1, stream2;
  if ((int)dcols / 64) {
    cudaStreamCreate(&stream2);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 64);
        dim3 blockDim(32, 8);
    aggregate_node_mul_sum_coarse2_kernel1<<<gridDim, blockDim, 0, stream2>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols);
  if ((dcols % 64 ) > 32) {
    cudaStreamCreate(&stream1);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
    dim3 blockDim(32, 8);
    aggregate_node_mul_sum_coarse2_kernel0_offset<<<gridDim, blockDim, 0, stream1>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /64) * 64);
  if ((dcols % 32 ) > 0) {
    cudaStreamCreate(&stream0);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
    dim3 blockDim(dcols %32, 8);
    aggregate_node_mul_sum_coarse2_kernel0_offset<<<gridDim, blockDim, 0, stream0>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /32) * 32);
  }
  }
else {
  if ((dcols % 64 ) > 0) {
    cudaStreamCreate(&stream0);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
    dim3 blockDim(dcols %64, 8);
    aggregate_node_mul_sum_coarse2_kernel0_offset<<<gridDim, blockDim, 0, stream0>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /64) * 64);
  }
}
  }
else {
  if ((int)dcols / 32) {
    cudaStreamCreate(&stream1);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 32);
        dim3 blockDim(32, 8);
    aggregate_node_mul_sum_coarse2_kernel0<<<gridDim, blockDim, 0, stream1>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols);
  if ((dcols % 32 ) > 0) {
    cudaStreamCreate(&stream0);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
    dim3 blockDim(dcols %32, 8);
    aggregate_node_mul_sum_coarse2_kernel0_offset<<<gridDim, blockDim, 0, stream0>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /32) * 32);
  }
  }
else {
  if ((int)dcols) {
    cudaStreamCreate(&stream0);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
        dim3 blockDim((int)dcols, 8);
    aggregate_node_mul_sum_coarse2_kernel0<<<gridDim, blockDim, 0, stream0>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)], iden_ptr, &col_ptr[start_vals], nrows, dcols);
  }
}
}
}return output_dense;
}
class aggregate_node_mul_sum_coarse2_AutoGrad : public torch::autograd::Function<aggregate_node_mul_sum_coarse2_AutoGrad> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                     torch::Tensor input_dense, int li) {
            ctx->saved_data["li"] = li;
            torch::Tensor offset_graph = global_offset_graph[2 * li];
            torch::Tensor columns_graph = global_columns_graph[2 * li];
            torch::Tensor value_graph = global_value_graph[2 * li];
        torch::Tensor bounds = global_bounds[2 * li];
            int segments = global_segments[2 * li];
             return aggregate_node_mul_sum_coarse2_call(input_dense, offset_graph, columns_graph,
                                value_graph, bounds, segments);
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
            return {aggregate_node_mul_sum_coarse2_call(input_dense, offset_graph, columns_graph, value_graph, bounds, segments), torch::Tensor()};        }
    };
struct GALAGNN : torch::nn::Module {
torch::nn::Linear fc0{nullptr};
torch::nn::Linear fc1{nullptr};
GALAGNN(int size0, int size1, int size2){
fc0 = register_module("fc0", torch::nn::Linear(size0, size1));
fc1 = register_module("fc1", torch::nn::Linear(size1, size2));
   }
std::vector<torch::Tensor>
forward(torch::Tensor t_iden, int ep, int mod_v){

    torch::Tensor ones;
    torch::Tensor degrees;
    torch::Tensor norm;
    torch::Tensor res;
    auto options_ones = torch::TensorOptions()
                       .dtype(torch::kFloat)
                       .requires_grad(false)
                       .device(torch::kCUDA, 0);
ones = torch::ones({global_nrows, 1}, options_ones);
            torch::Tensor offset_graph_ones = global_offset_graph[2 * 0];
          torch::Tensor columns_graph_ones = global_columns_graph[2 * 0];
          torch::Tensor value_graph_ones = global_value_graph[2 * 0];
        torch::Tensor bounds_ones = global_bounds[2 * 0];
        int segments_ones = global_segments[2 * 0];
        degrees = aggregate_node_mul_sum_direct_coarse2_call(ones, offset_graph_ones, columns_graph_ones,
                            value_graph_ones, bounds_ones, segments_ones);

        norm = torch::pow(degrees, -0.500000);
res = fc0->forward(t_iden);
        res = norm * res;
    if (ep % mod_v == 0) {
      res = aggregate_node_mul_sum_coarse2_AutoGrad::apply(res, 0);
    } else {
      res = aggregate_node_mul_sum_coarse2_AutoGrad::apply(res, 0);
    }
        res = norm * res;
        res = torch::relu(res);
        res = norm * res;
    if (ep % mod_v == 0) {
      res = aggregate_node_mul_sum_coarse2_AutoGrad::apply(res, 0);
    } else {
      res = aggregate_node_mul_sum_coarse2_AutoGrad::apply(res, 0);
    }
        res = norm * res;
res = fc1->forward(res);
return {res};
    }
};
int main(int argc, char **argv) {
  typedef typename SM::itype iT;
  typedef typename SM::ntype nT;
  typedef typename SM::vtype vT;

  typedef typename DM::itype diT;
  typedef typename DM::ntype dnT;
  typedef typename DM::vtype dvT;
  auto options_int_tile = 
    torch::TensorOptions().dtype(torch::kInt).requires_grad(false);
  auto options_float_tile = 
    torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);

    SM adj0;
    std::string filename = "../../Data/papers100M_20/";
    readSM_npy32<SM>(filename, &adj0);

    // Adj info
    iT nrows = adj0.nrows();
    global_nrows = nrows;
    iT ncols = adj0.ncols();
    nT nvals0 = adj0.nvals();

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
    int classes =
    *std::max_element(labels.vals_ptr(), labels.vals_ptr() + labels.nvals()) + 1;
    global_classes = classes;
    global_emb_size = emb_size;
  std::vector<SM *> tiled_graph_tile;
      tiled_graph_tile.push_back(&adj0);
      torch::Tensor total_offsets_graph_tile;
      torch::Tensor total_cols_graph_tile;
      torch::Tensor total_vals_graph_tile;
      torch::Tensor total_bounds_graph_tile;
      std::vector<iT> tile_offsets_graph_tile =
        static_ord_col_breakpoints<SM>(&adj0, 1000000000.000000);
      iT segments_graph_tile = tile_offsets_graph_tile.size() - 1;
      total_offsets_graph_tile = torch::zeros({(adj0.nrows() + 1) * (segments_graph_tile)}, options_int_tile);
      total_cols_graph_tile = torch::zeros({adj0.nvals()}, options_int_tile);
      total_vals_graph_tile = torch::zeros({adj0.nvals()}, options_float_tile);
      total_bounds_graph_tile = torch::zeros({2 * (segments_graph_tile)}, options_int_tile);
      ord_col_tiling_torch(tile_offsets_graph_tile, total_offsets_graph_tile, total_cols_graph_tile, total_vals_graph_tile,
        total_bounds_graph_tile, &adj0);
      iT *offset_ptr_graph_tile = total_offsets_graph_tile.data_ptr<iT>();
      iT *col_ptr_graph_tile = total_cols_graph_tile.data_ptr<iT>();
      vT *val_ptr_graph_tile = total_vals_graph_tile.data_ptr<vT>();
  global_segments.push_back(segments_graph_tile);
  global_bounds.push_back(total_bounds_graph_tile);
  global_segments.push_back(segments_graph_tile);
  global_bounds.push_back(total_bounds_graph_tile);

  torch::Device device(torch::kCUDA);
  auto options_cu_int = torch::TensorOptions()
                            .dtype(torch::kInt)
                            .requires_grad(false)
                            .device(torch::kCUDA, 0);
  auto options_cu_float_grad = torch::TensorOptions()
                                   .dtype(torch::kFloat)
                                   .requires_grad(true)
                                   .device(torch::kCUDA, 0);
  auto options_cu_float_ngrad = torch::TensorOptions()
                                    .dtype(torch::kFloat)
                                    .requires_grad(false)
                                    .device(torch::kCUDA, 0);
  auto options_cu_bool = torch::TensorOptions()
                             .dtype(torch::kBool)
                             .requires_grad(false)
                             .device(torch::kCUDA, 0);
  auto options_cu_long =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA, 0);

int *dL;
  float *dB;
  bool *d_train_mask, *d_valid_mask, *d_test_mask;

  CUDA_CHECK(cudaMalloc((void **)&dB, (nrows * emb_size) * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&dL, nrows * sizeof(long)));
  CUDA_CHECK(cudaMalloc((void **)&d_train_mask, nrows * sizeof(bool)));
  CUDA_CHECK(cudaMalloc((void **)&d_valid_mask, nrows * sizeof(bool)));
  CUDA_CHECK(cudaMalloc((void **)&d_test_mask, nrows * sizeof(bool)));

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

  torch::Tensor t_iden =
      torch::from_blob(dB, {nrows, emb_size}, options_cu_float_grad);
  torch::Tensor t_labs = torch::from_blob(dL, {nrows}, options_cu_long);

  torch::Tensor t_train_mask =
      torch::from_blob(d_train_mask, {nrows}, options_cu_bool);
  torch::Tensor t_valid_mask =
      torch::from_blob(d_valid_mask, {nrows}, options_cu_bool);
  torch::Tensor t_test_mask =
      torch::from_blob(d_test_mask, {nrows}, options_cu_bool);


  int *dA_csrOffsets0, *dA_columns0; 
  float *dA_values0;

  CUDA_CHECK(cudaMalloc((void **)&dA_columns0, nvals0 * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&dA_values0, nvals0 * sizeof(float)));

  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets0, (nrows + 1) * segments_graph_tile * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(dA_csrOffsets0, offset_ptr_graph_tile,
                        (nrows + 1) * segments_graph_tile * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_columns0, col_ptr_graph_tile, nvals0 * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_values0, val_ptr_graph_tile, nvals0 * sizeof(float),
                        cudaMemcpyHostToDevice));
  torch::Tensor t_offsets0 =
      torch::from_blob(dA_csrOffsets0, {(nrows+ 1) * segments_graph_tile}, options_cu_int);
  torch::Tensor t_cols0 = torch::from_blob(dA_columns0, {nvals0}, options_cu_int);

  torch::Tensor t_vals0 =
      torch::from_blob(dA_values0, {nvals0}, options_cu_float_ngrad);
  global_offset_graph.push_back(t_offsets0);
  global_columns_graph.push_back(t_cols0);
  global_value_graph.push_back(t_vals0);
  global_offset_graph.push_back(t_offsets0);
    global_columns_graph.push_back(t_cols0);
    global_value_graph.push_back(t_vals0);











int num_iters = 100;
auto net = std::make_shared<GALAGNN>(128, 32, 172);net->to(device);torch::optim::Adam optimizer(
    net->parameters(), torch::optim::AdamOptions(0.010000).weight_decay(5e-4));
 int mod_v = 1;
 int skip_cache_warmup = 5;
 double start, end;
  double start_train, end_train;
  std::vector<double> times_arr, times_arr_train;
 for (size_t epoch = 1; epoch <= num_iters; ++epoch) {
    // Reset gradients.
    optimizer.zero_grad();
    // Execute the model on the input data.
    cudaDeviceSynchronize();
    start = get_time();
    torch::Tensor prediction =
        net->forward(t_iden, epoch, mod_v)[0];
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
    if (epoch >= skip_cache_warmup) {
      times_arr.push_back(end - start);
      times_arr_train.push_back(end_train - start_train);
    }
  }
 CUDA_CHECK(cudaFree(dB));
  std::cout << calc_mean(times_arr) << ","
            << calc_mean(times_arr) + calc_mean(times_arr_train) << std::endl;

}
