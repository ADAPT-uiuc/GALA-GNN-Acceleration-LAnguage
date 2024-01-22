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
//    float* oden_ptr = output_dense.data_ptr<float>();
//    std::vector<float> oden_array(oden_ptr, oden_ptr + full_iden);

//    std::vector<float> oden_array(full_iden, 0);
//    float oden_array[full_iden] = { 0 };
    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);
    auto output_dense = torch::zeros({nrows, dcols} , options);
    float *oden_array = output_dense.data_ptr<float>();

    int64_t *offset_ptr = offset_graph.data_ptr<int64_t>();

    int32_t *col_ptr = columns_graph.data_ptr<int32_t>();

    float *val_ptr = value_graph.data_ptr<float>();

#pragma omp parallel for schedule(static, 4)
    for (int32_t i = 0; i < nrows; i++) {
        float* temp = new float[dcols] ();

        for (int64_t e = offset_ptr[i]; e < offset_ptr[i + 1]; e++) {
            int32_t v = col_ptr[e];
            float val = val_ptr[e];

            for (int k = 0; k < dcols; k++) {
                temp[k] += (val * iden_ptr[v * dcols + k]);

            }
        }
        for (int k = 0; k < dcols; k++) {
            oden_array[i * dcols + k] += temp[k];
        }
    }
    // TODO 3 Check the memory consumption of these

    // TODO 2 Add the matrix multiplication needed by weight update
//    auto update_dense = torch::matmul(output_dense, weights);

    // TODO 4 need to pass all intermediate results. But fine for now since we are only
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
////    float* oden_ptr = output_dense.data_ptr<float>();
////    std::vector<float> oden_array(oden_ptr, oden_ptr + full_iden);
//
////    std::vector<float> oden_array(full_iden, 0);
////    float oden_array[full_iden] = { 0 };
//    float* oden_array = new float[full_iden] ();
////    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);
////    auto output_dense = torch::zeros({nrows, dcols} , options);
////    float *oden_array = output_dense.data_ptr<float>();
//
//    int64_t *offset_ptr = offset_graph.data_ptr<int64_t>();
//
//    int32_t *col_ptr = columns_graph.data_ptr<int32_t>();
//
//    float *val_ptr = value_graph.data_ptr<float>();
//
//#pragma omp parallel for schedule(static, 4)
//    for (int32_t i = 0; i < nrows; i++) {
//        for (int64_t e = offset_ptr[i]; e < offset_ptr[i + 1]; e++) {
//            int32_t v = col_ptr[e];
//            float val = val_ptr[e];
//
//            for (int k = 0; k < dcols; k++) {
//                oden_array[i * dcols + k] += (val * iden_ptr[v * dcols + k]);
//
//            }
//        }
//    }
//
////    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);
////    auto output_dense = torch::zeros({nrows, dcols} , options);
//    torch::Tensor output_dense = torch::from_blob(oden_array, {nrows, dcols}, torch::kFloat32);
//    return {output_dense};
//
//}

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
////    float* oden_ptr = output_dense.data_ptr<float>();
////    std::vector<float> oden_array(oden_ptr, oden_ptr + full_iden);
//
////    std::vector<float> oden_array(full_iden, 0);
////    float oden_array[full_iden] = { 0 };
//    std::vector<float> oden_array(oden_arr);
//
//    int64_t *offset_ptr = offset_graph.data_ptr<int64_t>();
//
//    int32_t *col_ptr = columns_graph.data_ptr<int32_t>();
//
//    float *val_ptr = value_graph.data_ptr<float>();
//
//#pragma omp parallel for schedule(static, 4)
//    for (int32_t i = 0; i < nrows; i++) {
//        for (int64_t e = offset_ptr[i]; e < offset_ptr[i + 1]; e++) {
//            int32_t v = col_ptr[e];
//            float val = val_ptr[e];
//
//            for (int k = 0; k < dcols; k++) {
//                oden_array.at(i * dcols + k) += (val * iden_ptr[v * dcols + k]);
//
//            }
//        }
//    }
//
//    torch::Tensor output_dense = torch::from_blob(oden_array.data(), {nrows, dcols}, torch::kFloat32);
//    return {output_dense};
//}



std::vector<torch::Tensor> gather_backward(
        torch::Tensor grad_h,
        torch::Tensor offset_graph,
        torch::Tensor columns_graph,
        torch::Tensor value_graph,
        torch::Tensor weights) {

    auto nrows = offset_graph.numel() - 1;
    auto nvals = value_graph.numel();
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", &gather_forward, "Gather forward");
m.def("backward", &gather_backward, "Gather backward");
}
