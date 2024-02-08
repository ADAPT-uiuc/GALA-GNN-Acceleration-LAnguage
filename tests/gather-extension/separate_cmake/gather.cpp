//
// Created by damitha on 12/7/23.
//
//#include <torch/extension.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <parallel/algorithm>

//#include <ATen/ParallelOpenMP.h>
#include <omp.h>

torch::Tensor d_sigmoid(torch::Tensor z) {
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
    return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
    auto e = z.exp();
    auto mask = (alpha * (e - 1)) < 0;
    return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<at::Tensor> tiling_graph(
        int64_t num_cols,
        torch::Tensor offset_graph,
        torch::Tensor columns_graph,
        torch::Tensor value_graph) {
    typedef int32_t iT;
    typedef int64_t nT;
    typedef float vT;

    // Initial limits of the data
    auto nrows = offset_graph.numel() - 1;
    auto nvals = columns_graph.numel();
    int ntiles = ceil(((double) nrows) / num_cols);

    int64_t *offset_ptr = offset_graph.data_ptr<int64_t>();

    auto start_offset_ptr = (nT *) aligned_alloc(64, (nrows + 1) * sizeof(nT));
    std::copy(offset_ptr, offset_ptr + (nrows + 1), start_offset_ptr);


    int32_t *col_ptr = columns_graph.data_ptr<int32_t>();
    float *val_ptr = value_graph.data_ptr<float>();

    auto row_counter_vec = (int64_t *) aligned_alloc(64, (nrows) * sizeof(int64_t));
    auto row_dcsr_vec = (bool *) aligned_alloc(64, (nrows) * sizeof(bool));

    // Outputs
    auto optionsfloat = torch::TensorOptions().dtype(torch::kFloat).requires_grad(false);
    auto optionsint32 = torch::TensorOptions().dtype(torch::kInt32).requires_grad(false);
    auto optionsint64 = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);

    auto output_tile_offset = torch::zeros({ntiles + 1}, optionsint64);
    int64_t *out_tile_offset_array = output_tile_offset.data_ptr<int64_t>();
    output_tile_offset[0] = 0;

    auto output_columns = torch::zeros({nvals}, optionsint32);
    int32_t *out_columns_array = output_columns.data_ptr<int32_t>();

    auto output_values = torch::zeros({nvals}, optionsfloat);
    float *out_values_array = output_values.data_ptr<float>();

    // Unknown
    std::vector<int64_t> vec_offsets;
    vec_offsets.push_back(0);
    std::vector<int_fast32_t> vec_rows;
    nT prev_offset = 0;
    int64_t new_nvals = 0;

    for (int nth_tile = 0; nth_tile < ntiles; nth_tile++) {
        int32_t j_start = nth_tile * num_cols;
        int32_t j_end = (nth_tile + 1) * num_cols;

#pragma omp parallel for schedule(dynamic, 4)
        for (int i_i = 0; i_i < nrows; i_i += 1) {
            int row_nvals = 0;

            int64_t first_node_edge = start_offset_ptr[i_i];
            int64_t last_node_edge = offset_ptr[i_i + 1];

            bool found_nnz = false;
            for (int64_t e = first_node_edge; e < last_node_edge; e++) {
                int32_t v = col_ptr[e];
                if (v >= j_start && v < j_end) {
                    found_nnz = true;
                    row_nvals += 1;
                } else if (v >= j_end) {
                    break;
                }
            }
            if (found_nnz) {
                row_counter_vec[i_i] = row_nvals;
                row_dcsr_vec[i_i] = true;
            } else {
                row_dcsr_vec[i_i] = false;
            }
        }

        iT new_nrows = __gnu_parallel::count(row_dcsr_vec, row_dcsr_vec + nrows, true);

        int64_t current_vec_offset = vec_offsets.size();
        int64_t current_vec_rows = vec_rows.size();

        vec_offsets.resize(vec_offsets.size() + new_nrows);
        vec_rows.resize(vec_rows.size() + new_nrows);
        out_tile_offset_array[nth_tile + 1] = out_tile_offset_array[nth_tile] + new_nrows;

        iT r_i = 0;
        for (iT ith_row = 0; ith_row < nrows; ith_row++) {
            if (row_dcsr_vec[ith_row]) {
                new_nvals += row_counter_vec[ith_row];
                vec_rows[current_vec_rows + r_i] = ith_row;
                vec_offsets[current_vec_offset + r_i++] = new_nvals;
            }
        }

        // Point to insert is the largest offset of the previous round
        // Don't need to do all rows. Just do the ones with values.
#pragma omp parallel for schedule(dynamic, 4)
        for (iT i_r = 0; i_r < new_nrows; i_r++) {
            iT node_i = vec_rows[current_vec_rows + i_r];

            nT first_node_edge = vec_offsets[current_vec_rows + i_r];
            nT last_node_edge = vec_offsets[current_vec_rows + i_r + 1];
            nT nr_vals = last_node_edge - first_node_edge;

            nT src_first_edge = start_offset_ptr[node_i];

            for (nT e = 0; e < nr_vals; e++) {
                nT dst_node = first_node_edge + e;
                nT src_node = src_first_edge + e;
                out_columns_array[dst_node] = col_ptr[src_node];
                out_values_array[dst_node] = val_ptr[src_node];
            }
            start_offset_ptr[node_i] = start_offset_ptr[node_i] + nr_vals;
        }

    }

    // Unknown when staring building the output
    auto output_offset = torch::zeros({(long)vec_offsets.size()}, optionsint64);
    int64_t *out_offset_array = output_offset.data_ptr<int64_t>();
    std::copy(vec_offsets.begin(), vec_offsets.end(), out_offset_array);

    auto output_rows = torch::zeros({(long)vec_rows.size()}, optionsint32);
    int32_t *out_rows_array = output_rows.data_ptr<int32_t>();
    std::copy(vec_rows.begin(), vec_rows.end(), out_rows_array);

    return {output_tile_offset, output_offset, output_rows, output_columns, output_values};
}


std::vector<at::Tensor> gather_forward(
        torch::Tensor input_dense,
        torch::Tensor offset_graph,
        torch::Tensor columns_graph,
        torch::Tensor value_graph,
        torch::Tensor weights,
        torch::Tensor bias) {
    // Initial limits of the data
    auto nrows = offset_graph.numel() - 1;
    auto nvals = columns_graph.numel();
    auto full_iden = input_dense.numel();
    auto dcols = full_iden / nrows;

    float *iden_ptr = input_dense.data_ptr<float>();
//    float* oden_ptr = output_dense.data_ptr<float>();
//    std::vector<float> oden_array(oden_ptr, oden_ptr + full_iden);

//    std::vector<float> oden_array(full_iden, 0);
//    float oden_array[full_iden] = { 0 };
    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);
    auto output_dense = torch::zeros({nrows, dcols}, options);
    float *oden_array = output_dense.data_ptr<float>();

    int64_t *offset_ptr = offset_graph.data_ptr<int64_t>();

    int32_t *col_ptr = columns_graph.data_ptr<int32_t>();

    float *val_ptr = value_graph.data_ptr<float>();

#pragma omp parallel for schedule(static, 4)
    for (int32_t i = 0; i < nrows; i++) {
        for (int64_t e = offset_ptr[i]; e < offset_ptr[i + 1]; e++) {
            int32_t v = col_ptr[e];
            float val = val_ptr[e];

            for (int k = 0; k < dcols; k++) {
                oden_array[i * dcols + k] += (val * iden_ptr[v * dcols + k]);

            }
        }
    }

    // #pragma omp parallel for schedule(static, 4)
//     for (int32_t i = 0; i < nrows; i++) {
//         auto tempArr = (float *) aligned_alloc(64, sizeof(float) * dcols);
//         std::memset(tempArr, 0, sizeof(float) * dcols);

//         for (int64_t e = offset_ptr[i]; e < offset_ptr[i + 1]; e++) {
//             int32_t v = col_ptr[e];
//             float val = val_ptr[e];

//             for (int k = 0; k < dcols; k++) {
//                 tempArr[k] += (val * iden_ptr[v * dcols + k]);

//             }
//         }
//         for (int k = 0; k < dcols; k++) {
//             oden_array[i * dcols + k] += tempArr[k];
//         }
//     }
    // TODO 3 Check the memory consumption of these

    // TODO 2 Add the matrix multiplication needed by weight update
//    auto update_dense = torch::matmul(output_dense, weights);

    // TODO 4 need to pass all intermediate results. But fine for now since we are only
    //  computing a single thing
    return {output_dense};
}

// std::vector<at::Tensor> gather_forward(
//         torch::Tensor input_dense,
//         torch::Tensor offset_graph,
//         torch::Tensor columns_graph,
//         torch::Tensor value_graph,
//         torch::Tensor weights,
//         torch::Tensor bias
//         torch::Tensor output_dense,) {
//     // Initial limits of the data
//     auto nrows = offset_graph.numel() - 1;
//     auto nvals = value_graph.numel();
//     auto full_iden = input_dense.numel();
//     auto dcols = full_iden / nrows;

//     float *iden_ptr = input_dense.data_ptr<float>();
// //    float* oden_ptr = output_dense.data_ptr<float>();
// //    std::vector<float> oden_array(oden_ptr, oden_ptr + full_iden);

// //    std::vector<float> oden_array(full_iden, 0);
// //    float oden_array[full_iden] = { 0 };

//     float *oden_array = output_dense.data_ptr<float>();

//     int64_t *offset_ptr = offset_graph.data_ptr<int64_t>();

//     int32_t *col_ptr = columns_graph.data_ptr<int32_t>();

//     float *val_ptr = value_graph.data_ptr<float>();

// #pragma omp parallel for schedule(static, 4)
//     for (int32_t i = 0; i < nrows; i++) {
//         for (int64_t e = offset_ptr[i]; e < offset_ptr[i + 1]; e++) {
//             int32_t v = col_ptr[e];
//             float val = val_ptr[e];

//             for (int k = 0; k < dcols; k++) {
//                 oden_array[i * dcols + k] += (val * iden_ptr[v * dcols + k]);
//             }
//         }
//     }
//     // TODO 3 Check the memory consumption of these

//     // TODO 2 Add the matrix multiplication needed by weight update
// //    auto update_dense = torch::matmul(output_dense, weights);

//     // TODO 4 need to pass all intermediate results. But fine for now since we are only
//     //  computing a single thing
//     return {};
// }


// std::vector<at::Tensor> gather_forward(
//        torch::Tensor input_dense,
//        torch::Tensor offset_graph,
//        torch::Tensor columns_graph,
//        torch::Tensor value_graph,
//        torch::Tensor weights,
//        torch::Tensor bias) {
//    // Initial limits of the data
//    auto nrows = offset_graph.numel() - 1;
//    auto nvals = value_graph.numel();
//    auto full_iden = input_dense.numel();
//    auto dcols = full_iden / nrows;

//    float *iden_ptr = input_dense.data_ptr<float>();
// //    float* oden_ptr = output_dense.data_ptr<float>();
// //    std::vector<float> oden_array(oden_ptr, oden_ptr + full_iden);

// //    std::vector<float> oden_array(full_iden, 0);
// //    float oden_array[full_iden] = { 0 };
//     // TODO Try to move this allocation out of here.
//    float* oden_array = new float[full_iden] ();
// //    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);
// //    auto output_dense = torch::zeros({nrows, dcols} , options);
// //    float *oden_array = output_dense.data_ptr<float>();



//    int64_t *offset_ptr = offset_graph.data_ptr<int64_t>();

//    int32_t *col_ptr = columns_graph.data_ptr<int32_t>();

//    float *val_ptr = value_graph.data_ptr<float>();

// #pragma omp parallel for schedule(static, 4)
//    for (int32_t i = 0; i < nrows; i++) {
//        for (int64_t e = offset_ptr[i]; e < offset_ptr[i + 1]; e++) {
//            int32_t v = col_ptr[e];
//            float val = val_ptr[e];

//            for (int k = 0; k < dcols; k++) {
//                oden_array[i * dcols + k] += (val * iden_ptr[v * dcols + k]);

//            }
//        }
//    }

// //    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);
// //    auto output_dense = torch::zeros({nrows, dcols} , options);
//    torch::Tensor output_dense = torch::from_blob(oden_array, {nrows, dcols}, torch::kFloat32);
//    return {output_dense};

// }


std::vector<at::Tensor> gather_forward_tile(
        torch::Tensor input_dense,
        torch::Tensor tile_offset_graph,
        torch::Tensor offset_graph,
        torch::Tensor rows_graph,
        torch::Tensor columns_graph,
        torch::Tensor value_graph,
        torch::Tensor weights,
        torch::Tensor bias) {
    // Initial limits of the data
    auto ntiles = tile_offset_graph.numel() - 1;
    auto nrows = offset_graph.numel() - 1;
    auto nvals = columns_graph.numel();
    auto full_iden = input_dense.numel();
    auto dcols = full_iden / nrows;

    float *iden_ptr = input_dense.data_ptr<float>();

    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);
    auto output_dense = torch::zeros({nrows, dcols}, options);
    float *oden_array = output_dense.data_ptr<float>();

    int64_t *tile_offset_ptr = tile_offset_graph.data_ptr<int64_t>();
    int64_t *offset_ptr = offset_graph.data_ptr<int64_t>();
    int32_t *row_ptr = rows_graph.data_ptr<int32_t>();
    int32_t *col_ptr = columns_graph.data_ptr<int32_t>();
    float *val_ptr = value_graph.data_ptr<float>();

    // Iterate through the graph tiles
    for (int jj = 0; jj < ntiles; jj++) {
        // Get the relevant row offsets
        int64_t tile_offset_start = tile_offset_ptr[jj];
        int64_t tile_offset_end = tile_offset_ptr[jj + 1];

#pragma omp parallel for schedule(static, 4)
        for (int64_t i = tile_offset_start; i < tile_offset_end; i++) {
            int32_t u = row_ptr[i];
            for (int64_t e = offset_ptr[i]; e < offset_ptr[i + 1]; e++) {
                int32_t v = col_ptr[e];
                float val = val_ptr[e];

                for (int k = 0; k < dcols; k++) {
                    oden_array[u * dcols + k] += (val * iden_ptr[v * dcols + k]);
                }
            }
        }
    }

    // TODO 3 Check the memory consumption of these

    // TODO 2 Add the matrix multiplication needed by weight update
//    auto update_dense = torch::matmul(output_dense, weights);

    // TODO 4 need to pass all intermediate results. But fine for now since we are only
    //  computing a single thing
    return {output_dense};
}

std::vector<torch::Tensor> gather_backward(
        torch::Tensor grad_h,
        torch::Tensor offset_graph,
        torch::Tensor columns_graph,
        torch::Tensor value_graph,
        torch::Tensor weights) {

    auto nrows = offset_graph.numel() - 1;
    auto nvals = columns_graph.numel();
    auto full_iden = grad_h.numel();
    auto dcols = full_iden / nrows;

    float *grad_h_ptr = grad_h.data_ptr<float>();
//    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);
    auto d_input_dense = torch::zeros({nrows, dcols}, torch::kFloat);
    float *d_input_ptr = d_input_dense.data_ptr<float>();

    int64_t *offset_ptr = offset_graph.data_ptr<int64_t>();
    int32_t *col_ptr = columns_graph.data_ptr<int32_t>();
    float *val_ptr = value_graph.data_ptr<float>();

    // TODO Need to add getting the transpose of the graph

#pragma omp parallel for schedule(static, 4)
    for (int32_t i = 0; i < nrows; i++) {
        for (int64_t e = offset_ptr[i]; e < offset_ptr[i + 1]; e++) {
            int32_t v = col_ptr[e];
            float val = val_ptr[e];

            for (int k = 0; k < dcols; k++) {
                d_input_ptr[i * dcols + k] += (val * grad_h_ptr[v * dcols + k]);
            }
        }
    }
    return {d_input_dense};
}

//
//
//static auto registry =
//        torch::jit::RegisterOperators("gala::gather_forward", &gather_forward)
//        .op("gala::gather_backward", &gather_backward);

TORCH_LIBRARY(gala_ops, m) {
m.def("gather_forward", gather_forward);
m.def("gather_forward_tile", gather_forward_tile);
m.def("tiling_graph", tiling_graph);
}

