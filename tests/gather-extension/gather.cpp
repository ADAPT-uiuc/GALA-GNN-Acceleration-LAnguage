//
// Created by damitha on 12/7/23.
//
#include <torch/extension.h>

#include <iostream>

#include <vector>

#define _OPENMP
#include <ATen/ParallelOpenMP.h>

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

std::vector<at::Tensor> gather_forward(
        torch::Tensor input_dense,
        torch::Tensor offset_graph,
        torch::Tensor columns_graph,
        torch::Tensor value_graph,
        torch::Tensor weights,
        torch::Tensor bias) {
    // Initial limits of the data
    auto nrows = offset_graph.numel() - 1;
    auto nvals = value_graph.numel();
    auto full_iden = input_dense.numel();
    auto dcols = full_iden / nrows;

    float *iden_ptr = input_dense.data_ptr<float>();
//    std::vector<float> iden_array(iden_ptr, iden_ptr + full_iden);

//    float* oden_ptr = output_dense.data_ptr<float>();
//    std::vector<float> oden_array(oden_ptr, oden_ptr + full_iden);

    std::vector<float> oden_array(full_iden, 0.0f);

    int64_t *offset_ptr = offset_graph.data_ptr<int64_t>();
//    std::vector<int64_t> offset_array(offset_ptr, offset_ptr + nrows + 1);

    int32_t *col_ptr = columns_graph.data_ptr<int32_t>();
//    std::vector<int32_t> col_array(col_ptr, col_ptr + nvals);

    float *val_ptr = value_graph.data_ptr<float>();
//    std::vector<float> val_array(val_ptr, val_ptr + nvals);

#pragma omp parallel for schedule(static, 1)
    for (int32_t i = 0; i < nrows; i++) {
        for (int64_t e = offset_ptr[i]; e < offset_ptr[i + 1]; e++) {
            int32_t v = col_ptr[e];
            float val = val_ptr[e];

            for (int k = 0; k < dcols; k++) {
                oden_array[i * dcols + k] += (val * iden_ptr[v * dcols + k]);
            }
        }
    }
    // TODO 3 Check the memory consumption of these

    // TODO 2 Add the matrix multiplication needed by weight update

    // TODO 1 see if you get a proper output from this or if you need a conversion at some point
    auto output_dense = torch::from_blob(oden_array.data(), {oden_array.size()}, torch::kFloat);
    // TODO 4 need to pass all intermediate results. But finr for now since we are only
    //  computing a single thing
    return {output_dense};
}

//std::vector<at::Tensor> gather_forward(
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
//
//    float *iden_ptr = input_dense.data_ptr<float>();
////    std::vector<float> iden_array(iden_ptr, iden_ptr + full_iden);
//
////    float* oden_ptr = output_dense.data_ptr<float>();
////    std::vector<float> oden_array(oden_ptr, oden_ptr + full_iden);
//
//    std::vector<float> oden_array(full_iden, 0.0f);
//
//    int64_t *offset_ptr = offset_graph.data_ptr<int64_t>();
////    std::vector<int64_t> offset_array(offset_ptr, offset_ptr + nrows + 1);
//
//    int32_t *col_ptr = columns_graph.data_ptr<int32_t>();
////    std::vector<int32_t> col_array(col_ptr, col_ptr + nvals);
//
//    float *val_ptr = value_graph.data_ptr<float>();
////    std::vector<float> val_array(val_ptr, val_ptr + nvals);
//
//    at::parallel_for(0, nrows, 16, [&](int32_t start, int32_t end) {
////        for (int64_t b = start; b < end; b++)
////        {
////            z_out[b] = z[b] * z[b];
////        }
////        std::cout << "hi there from " << omp_get_thread_num() << std::endl;
//        for (int32_t i = start; i < end; i++) {
//            for (int64_t e = offset_ptr[i]; e < offset_ptr[i + 1]; e++) {
//                int32_t v = col_ptr[e];
//                float val = val_ptr[e];
//
//                for (int k = 0; k < dcols; k++) {
//                    oden_array[i * dcols + k] += (val * iden_ptr[v * dcols + k]);
//                }
//            }
//        }
//    });
//
////#pragma omp parallel for schedule(static, 1)
////    for (int32_t i = 0; i < nrows; i++) {
////        for (int64_t e = offset_ptr[i]; e < offset_ptr[i + 1]; e++) {
////            int32_t v = col_ptr[e];
////            float val = val_ptr[e];
////
////            for (int k = 0; k < dcols; k++) {
////                oden_array[i * dcols + k] += (val * iden_ptr[v * dcols + k]);
////            }
////        }
////    }
//    // TODO 3 Check the memory consumption of these
//
//    // TODO 2 Add the matrix multiplication needed by weight update
//
//    // TODO 1 see if you get a proper output from this or if you need a conversion at some point
//    auto output_dense = torch::from_blob(oden_array.data(), {oden_array.size()}, torch::kFloat);
//    // TODO 4 need to pass all intermediate results. But finr for now since we are only
//    //  computing a single thing
//    return {output_dense};
//}

std::vector<torch::Tensor> gather_backward(
        torch::Tensor grad_h,
        torch::Tensor weights) {
//    auto d_output_gate = torch::tanh(new_cell) * grad_h;
//    auto d_tanh_new_cell = output_gate * grad_h;
//    auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;
//
//    auto d_old_cell = d_new_cell;
//    auto d_candidate_cell = input_gate * d_new_cell;
//    auto d_input_gate = candidate_cell * d_new_cell;
//
//    auto gates = gate_weights.chunk(3, /*dim=*/1);
//    d_input_gate *= d_sigmoid(gates[0]);
//    d_output_gate *= d_sigmoid(gates[1]);
//    d_candidate_cell *= d_elu(gates[2]);
//
//    auto d_gates =
//            torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);
//
//    auto d_weights = d_gates.t().mm(X);
//    auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);
//
//    auto d_X = d_gates.mm(weights);
//    const auto state_size = grad_h.size(1);
//    auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
//    auto d_input = d_X.slice(/*dim=*/1, state_size);

    return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", &gather_forward, "Gather forward");
m.def("backward", &gather_backward, "Gather backward");
}
