//
// Created by damitha on 5/12/24.
//
// Define a new Module.
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>

#include <torch/script.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <parallel/algorithm>

//#include <ATen/ParallelOpenMP.h>
#include <stdlib.h>
#include <omp.h>
#include <bits/stdc++.h>

typedef int ind1_t;
typedef int ind2_t;
typedef float val_t;

#include "../../src/utils/mtx_io.h"
#include "../../src/matrix/csrc_matrix.h"
#include "../../src/matrix/dense_matrix.h"
#include "../../src/ops/aggregators.h"
#include "../../src/ops/sparse_matrix_ops.h"
#include "../../src/ops/tiling.h"
#include "../common.h"
#include "util/cuda_util.cuh"

#include <torch/torch.h>

//Dense matrix with double values.
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

template <int CoarsenFactor>
__global__ void csrspmm_rowcaching_rowbalance_kernel(
        const int M, // rows 
        const int N, // 
        const int K, //
        const int dcsr_indptr[], // offsets
        const int dcsr_rows[], // cols
        const int dcsr_indices[], // cols
        const float dcsr_data[], // vals
        const float B[],
        float C[]) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    extern __shared__ int shared_mem[];
    int *workspace_indices = &shared_mem[(warp_id << 5)];
    float *workspace_data =
            (float *)(workspace_indices +
                      blockDim.x); // float and int has the same size

    // get the sparse-value range of this row
    int row_id_point = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (row_id_point >= M)
        return;
    // Get the working row
    int row_id = dcsr_rows[row_id_point];
    int start = dcsr_indptr[row_id];
    int end = dcsr_indptr[row_id + 1];

    // get the dense column offset
    int col_offset = blockIdx.y * 32 * CoarsenFactor;
    const float *B_lanes[CoarsenFactor];
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
        B_lanes[i] = B + col_offset + lane_id + i * 32;
    }
    int ldB = N;

    // declare accumulators
    float c[CoarsenFactor] = {0.0f};
    int ldC = N;

    // N-dimension residual handling
    if (blockIdx.y == gridDim.y - 1)
        goto Ndim_Residue;

    // iterate over the sparse row
    for (int p = start; p < end; p += 32) {
        // copy a bucket of sparse row elements into shared memory
        if (p + lane_id < end) {
            workspace_data[lane_id] =
                    __guard_load_default_one<float>(dcsr_data, (p + lane_id));
            workspace_indices[lane_id] = dcsr_indices[p + lane_id];
        } else {
            workspace_data[lane_id] = 0.0f;
            workspace_indices[lane_id] = 0;
        }
        __syncwarp();
// do MAC computation using buffered elements
#pragma unroll
        for (int pp = 0; pp < 32; pp++) {
            int k = workspace_indices[pp];
            float v = workspace_data[pp];
#pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                c[i] += v * B_lanes[i][k * ldB];
            }
        }
    }

// write results
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
        float *C_lane = B_lanes[i] - B + (C + row_id * ldC);
        *C_lane = c[i];
    }
    return;

    Ndim_Residue:
    int valid_lane_num = CEIL(N - col_offset - lane_id, 32);

    // iterate over the sparse row
    for (int p = start; p < end; p += 32) {
        // copy a bucket of sparse row elements into shared memory
        if (p + lane_id < end) {
            workspace_data[lane_id] =
                    __guard_load_default_one<float>(dcsr_data, (p + lane_id));
            workspace_indices[lane_id] = dcsr_indices[p + lane_id];
        } else {
            workspace_data[lane_id] = 0.0f;
            workspace_indices[lane_id] = 0;
        }
        __syncwarp();
// do MAC computation using buffered elements
#pragma unroll
        for (int pp = 0; pp < 32; pp++) {
            int k = workspace_indices[pp];
            float v = workspace_data[pp];
#pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                if (i < valid_lane_num) {
                    c[i] += v * B_lanes[i][k * ldB];
                }
            }
        }
    }

// write results
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
        float *C_lane = B_lanes[i] - B + (C + row_id * ldC);
        if (i < valid_lane_num) {
            *C_lane = c[i];
        }
    }
    return;
}


//template <int CoarsenFactor, int ThreadNz>
//__global__ void csrspmm_rowcaching_nnzbalance_kernel(
//        const int M, const int N, const int K, const int nnz_,
//        const int csr_indptr[], const int csr_indices[], const float csr_data[],
//        const float B[], float C[]) {
//    int nnz = nnz_;
//    if (nnz < 0)
//        nnz = csr_indptr[M];
//
//    int warp_id = threadIdx.x >> 5;
//    int lane_id = threadIdx.x & 31;
//
//    extern __shared__ int shared_mem[];
//    int *workspace_rowid = &shared_mem[(warp_id << 5)];
//    int *workspace_colid = workspace_rowid + blockDim.x;
//    float *workspace_data =
//            (float *)(workspace_colid +
//                      blockDim.x); // float and int has the same size
//
//    // get the sparse-value range of this row
//    int global_warp_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
//    int nz_start = global_warp_id * (ThreadNz * 32);
//
//    // get the dense column offset
//    int col_offset = blockIdx.y * 32 * CoarsenFactor;
//    const float *B_lanes[CoarsenFactor];
//    float *C_lanes[CoarsenFactor];
//#pragma unroll
//    for (int i = 0; i < CoarsenFactor; i++) {
//        B_lanes[i] = B + col_offset + lane_id + i * 32;
//        C_lanes[i] = C + col_offset + lane_id + i * 32;
//    }
//    int ldB = N;
//
//    // declare accumulators
//    float c[CoarsenFactor] = {0.0f};
//    int ldC = N;
//
//    int stride = gridDim.x * (blockDim.x >> 5) * ThreadNz * 32;
//
//    if (blockIdx.y == gridDim.y - 1)
//        goto Ndim_Residue;
//
//    for (; nz_start < nnz; nz_start += stride) {
//        // iterate over the segment of this warp
//        for (int tile_base = nz_start;
//             tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {
//            int thread_nz_id = tile_base + lane_id;
//            if (thread_nz_id < nnz) {
//                workspace_colid[lane_id] = csr_indices[thread_nz_id];
//                workspace_data[lane_id] =
//                        __guard_load_default_one<float>(csr_data, thread_nz_id);
//            } else {
//                workspace_colid[lane_id] = 0;
//                workspace_data[lane_id] = 0.0f;
//            }
//            workspace_rowid[lane_id] =
//                    binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
//            __syncwarp();
//
//            // initialize with first value
//            int k = workspace_colid[0];
//            float v = workspace_data[0];
//#pragma unroll
//            for (int i = 0; i < CoarsenFactor; i++) {
//                c[i] = v * B_lanes[i][k * ldB];
//            }
//            int row_curr = workspace_rowid[0], next_row;
//
//// scan
//#pragma unroll
//            for (int pp = 1; pp < 32; pp++) {
//                next_row = workspace_rowid[pp];
//                if (next_row != row_curr) {
//#pragma unroll
//                    for (int i = 0; i < CoarsenFactor; i++) {
//                        atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
//                    }
//                    row_curr = next_row;
//                    k = workspace_colid[pp];
//                    v = workspace_data[pp];
//#pragma unroll
//                    for (int i = 0; i < CoarsenFactor; i++) {
//                        c[i] = v * B_lanes[i][k * ldB];
//                    }
//                } else {
//                    k = workspace_colid[pp];
//                    v = workspace_data[pp];
//#pragma unroll
//                    for (int i = 0; i < CoarsenFactor; i++) {
//                        c[i] = c[i] + v * B_lanes[i][k * ldB];
//                    }
//                }
//            }
//#pragma unroll
//            for (int i = 0; i < CoarsenFactor; i++) {
//                atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
//            }
//        }
//    }
//    return;
//
//    Ndim_Residue:
//
//    int valid_lane_num = CEIL(N - col_offset - lane_id, 32);
//
//    for (; nz_start < nnz; nz_start += stride) {
//        // iterate over the segment of this warp
//        for (int tile_base = nz_start;
//             tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {
//            int thread_nz_id = tile_base + lane_id;
//            if (thread_nz_id < nnz) {
//                workspace_colid[lane_id] = csr_indices[thread_nz_id];
//                workspace_data[lane_id] =
//                        __guard_load_default_one<float>(csr_data, thread_nz_id);
//            } else {
//                workspace_colid[lane_id] = 0;
//                workspace_data[lane_id] = 0.0f;
//            }
//            workspace_rowid[lane_id] =
//                    binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
//            __syncwarp();
//
//            // initialize with first value
//            int k = workspace_colid[0];
//            float v = workspace_data[0];
//#pragma unroll
//            for (int i = 0; i < CoarsenFactor; i++) {
//                if (i < valid_lane_num) {
//                    c[i] = v * B_lanes[i][k * ldB];
//                }
//            }
//            int row_curr = workspace_rowid[0], next_row;
//
//// scan
//#pragma unroll
//            for (int pp = 1; pp < 32; pp++) {
//                next_row = workspace_rowid[pp];
//                if (next_row != row_curr) {
//#pragma unroll
//                    for (int i = 0; i < CoarsenFactor; i++) {
//                        if (i < valid_lane_num) {
//                            atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
//                        }
//                    }
//                    row_curr = next_row;
//                    k = workspace_colid[pp];
//                    v = workspace_data[pp];
//#pragma unroll
//                    for (int i = 0; i < CoarsenFactor; i++) {
//                        if (i < valid_lane_num) {
//                            c[i] = v * B_lanes[i][k * ldB];
//                        }
//                    }
//                } else {
//                    k = workspace_colid[pp];
//                    v = workspace_data[pp];
//#pragma unroll
//                    for (int i = 0; i < CoarsenFactor; i++) {
//                        if (i < valid_lane_num) {
//                            c[i] = c[i] + v * B_lanes[i][k * ldB];
//                        }
//                    }
//                }
//            }
//#pragma unroll
//            for (int i = 0; i < CoarsenFactor; i++) {
//                if (i < valid_lane_num) {
//                    atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
//                }
//            }
//        }
//    }
//}


std::vector <at::Tensor> gather_forward_gcn(
        torch::Tensor input_dense,
        torch::Tensor offset_graph,
        torch::Tensor rows_graph,
        torch::Tensor columns_graph,
        torch::Tensor value_graph,
        torch::Tensor ranges,
        int nrows, int segments) {
    auto full_iden = input_dense.numel();
    auto dcols = full_iden / nrows;

    // // Dense
    // Input
    float *iden_ptr = input_dense.data_ptr<float>();
    // Output
    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true).device(torch::kCUDA, 0);
    auto output_dense = torch::zeros({nrows, dcols}, options);
    float *oden_array = output_dense.data_ptr<float>();

    // Sparse
    int *offset_ptr = offset_graph.data_ptr<int>();
    int *row_ptr = rows_graph.data_ptr<int>(); 
    int *col_ptr = columns_graph.data_ptr<int>();
    float *val_ptr = value_graph.data_ptr<float>();
    int *ranges_ptr = ranges.data_ptr<int>();

    float alpha = 1.0f;
    float beta = 1.0f;

    for (int i = 0; i < segments; i++){
        int i1 = i;
        int start_rows = ranges_ptr[i1 * 2];
        int end_rows = ranges_ptr[i1 * 2 + 1];
        int nrows_seg = end_rows - start_rows;

        // factor of thread coarsening
        int coarsen_factor = (dcols >= 128) ? 4 : (dcols >= 64) ? 2 : 1;
        // number of parallel warps along M-dimension
        int Mdim_worker = nrows;
        // partition large-N and map to blockdim.y to help cache performance
        int Ndim_threadblock = CEIL(dcols, coarsen_factor);

        int ref_warp_per_tb = RefThreadPerBlock / 32;
        int Mdim_warp_per_tb = ref_warp_per_tb;

        // total number of warps
        int gridDimX = CEIL(Mdim_worker, Mdim_warp_per_tb);
        int gridDimY = Ndim_threadblock;
        dim3 gridDim(gridDimX, gridDimY, 1);
        dim3 blockDim(32, Mdim_warp_per_tb, 1);

        // TODO: Need to change the offset start points etc. col and vals do not need + start_vals.
        if (coarsen_factor == 4) {
            csrspmm_rowcaching_rowbalance_kernel<4>
            <<<gridDim, blockDim>>>(nrows_seg, 
                                    dcols, 
                                    nrows, 
                                    offset_ptr + start_rows,
                                    row_ptr + start_rows,
                                    col_ptr,
                                    val_ptr, 
                                    iden_ptr,
                                    oden_array);
    //        csrspmm_non_transpose_parreduce_rowbalance_kernel<4>
    //        <<<gridDim, blockDim>>>(nrows, dcols, nrows, offset_ptr,
    //                                col_ptr, val_ptr, iden_ptr, oden_array);
        } else if (coarsen_factor == 2) {
            csrspmm_rowcaching_rowbalance_kernel<2>
            <<<gridDim, blockDim>>>(nrows_seg, 
                                    dcols, 
                                    nrows, 
                                    offset_ptr + start_rows,
                                    row_ptr + start_rows,
                                    col_ptr,
                                    val_ptr, 
                                    iden_ptr,
                                    oden_array);
    //        csrspmm_non_transpose_parreduce_rowbalance_kernel<2>
    //        <<<gridDim, blockDim>>>(nrows, dcols, nrows, offset_ptr,
    //                                col_ptr, val_ptr, iden_ptr, oden_array);
        } else {
            csrspmm_rowcaching_rowbalance_kernel<1>
            <<<gridDim, blockDim>>>(nrows_seg, 
                                    dcols, 
                                    nrows, 
                                    offset_ptr + start_rows,
                                    row_ptr + start_rows,
                                    col_ptr,
                                    val_ptr, 
                                    iden_ptr,
                                    oden_array);
    //        csrspmm_non_transpose_parreduce_rowbalance_kernel<1>
    //        <<<gridDim, blockDim>>>(nrows, dcols, nrows, offset_ptr,
    //                                col_ptr, val_ptr, iden_ptr, oden_array);
        }
    }

    return {output_dense};
}


struct GCN : torch::nn::Module {

    // Implement the Net's algorithm.
    std::vector<torch::Tensor> forward(torch::Tensor input_dense,
                                       torch::Tensor offset_graph,
                                       torch::Tensor rows_graph,
                                       torch::Tensor columns_graph,
                                       torch::Tensor value_graph,
                                       torch::Tensor ranges,
                                       int nrows, int segments) {
        return gather_forward_gcn(input_dense, offset_graph, rows_graph, columns_graph, value_graph, ranges, nrows, segments);
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
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

    std::vector<iT> tile_offsets = static_ord_col_breakpoints<SM>(&adj, cols_per_tile);

    iT segments = tile_offsets.size() - 1;

    ord_col_tiling_torch_dcsr(tile_offsets,
                              total_row_range, // Only CPU
                              total_rows, // All below are in both CPU and GPU
                              total_offsets,
                              total_cols,
                              total_vals,
                              &adj); // Except this.

//    int dcsr_nrows =  total_rows.sizes()[0];
//    int dcsr_noffsets = total_offsets.sizes()[0];
//
//    iT *row_ptr = total_rows.data_ptr<iT>();
//    iT *offset_ptr = total_offsets.data_ptr<iT>();
//    iT *col_ptr = total_cols.data_ptr<iT>();
//    vT *val_ptr = total_vals.data_ptr<vT>();
//
//    // Init input with random numbers
//    DM input_emb;
//    input_emb.build(adj.ncols(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
//    for (diT i = 0; i < adj.nrows(); i++) {
//        for (dnT j = 0; j < emb_size; j++) {
//            input_emb.vals_ptr()[i * emb_size + j] = (dvT) (rand() % 100) / 100;
//        }
//    }
//    input_emb.set_all(1);
//
//    DM out_emb;
//    out_emb.build(adj.nrows(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
//
//    DM out_emb2;
//    out_emb2.build(adj.nrows(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
//
//    auto wsum_aggr = wsumAgg<val_t, val_t, ind2_t>;
//
//    // Comparison for checking if SpMM works correctlu
//    out_emb.set_all(0);
//    out_emb2.set_all(0);
//
//    std::cout << adj.nrows() << " " << adj.ncols() << " " << adj.nvals() << std::endl;
//
//    //    gSpMM(&adj, &input_emb, &out_emb2, wsum_aggr);
//
//
//    // Create a new Net.
//    auto net = std::make_shared<GCN>();
//
//    // Instantiate an SGD optimization algorithm to update our Net's parameters.
//    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);
//
//    iT *dA_dcsrOffsets, *dA_columns, *dA_rows;
//    float *dA_values, *dB;
//
//    // Malloc on GPU
//    CUDA_CHECK(cudaMalloc((void **) &dA_dcsrOffsets,
//                          (dcsr_noffsets) * sizeof(iT)));
//    CUDA_CHECK(cudaMalloc((void **) &dA_rows,
//                          (dcsr_nrows) * sizeof(iT)));
//    CUDA_CHECK(cudaMalloc((void **) &dA_columns, nvals * sizeof(iT)));
//    CUDA_CHECK(cudaMalloc((void **) &dA_values, nvals * sizeof(vT)));
//    CUDA_CHECK(cudaMalloc((void **) &dB, (nrows * emb_size) * sizeof(vT)));
//
//    // Copy to GPU
//    CUDA_CHECK(cudaMemcpy(dA_dcsrOffsets, offset_ptr, (dcsr_noffsets) * sizeof(iT),
//                          cudaMemcpyHostToDevice));
//    CUDA_CHECK(cudaMemcpy(dA_rows, row_ptr, (dcsr_nrows) * sizeof(iT),
//                          cudaMemcpyHostToDevice));
//    CUDA_CHECK(cudaMemcpy(dA_columns, col_ptr, nvals * sizeof(iT),
//                          cudaMemcpyHostToDevice));
//    CUDA_CHECK(cudaMemcpy(dA_values, val_ptr, nvals * sizeof(float),
//                          cudaMemcpyHostToDevice));
//    CUDA_CHECK(cudaMemcpy(dB, input_emb.vals_ptr(), (nrows * emb_size)  * sizeof(float),
//                          cudaMemcpyHostToDevice));
//
//    // Create Torch tensors
//    // The graph inputs
//    auto options_cu_int = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, 0);
//    torch::Tensor t_offsets = torch::from_blob(dA_dcsrOffsets, {dcsr_noffsets}, options_cu_int);
//    torch::Tensor t_rows = torch::from_blob(dA_rows, {dcsr_nrows}, options_cu_int);
//    torch::Tensor t_cols = torch::from_blob(dA_columns, {nvals}, options_cu_int);
//
//    auto options_cu_float = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, 0);
//    torch::Tensor t_vals = torch::from_blob(dA_values, {nvals}, options_cu_float);
//    // The dense input
//    torch::Tensor t_iden = torch::from_blob(dB, {nrows * emb_size}, options_cu_float);
//
//    double start, end;
//    val_t randVal;
//    std::vector<double> times_arr;
//    for (size_t epoch = 1; epoch <= num_iters; ++epoch) {
//        // Reset gradients.
//        optimizer.zero_grad();
//        // Execute the model on the input data.
//        cudaDeviceSynchronize();
//        start = get_time();
//        torch::Tensor prediction = net->forward(t_iden,
//                                                t_offsets,
//                                                t_rows,
//                                                t_cols,
//                                                t_vals,
//                                                total_row_range,
//                                                nrows,
//                                                segments)[0];
//
//        cudaDeviceSynchronize();
//        end = get_time();
//
//        if (epoch >= skip_cache_warmup) {
//            times_arr.push_back(end - start);
//        }
//    }
//
//    CUDA_CHECK(cudaFree(dA_dcsrOffsets));
//    CUDA_CHECK(cudaFree(dA_rows));
//    CUDA_CHECK(cudaFree(dA_values));
//    CUDA_CHECK(cudaFree(dA_columns));
//    CUDA_CHECK(cudaFree(dB));
//
//    std::cout << calc_mean(times_arr) << "," << calc_std(times_arr) << std::endl;

}