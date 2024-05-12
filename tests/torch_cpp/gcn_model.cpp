//
// Created by damitha on 5/12/24.
//
// Define a new Module.
#include "gather.cu"

typedef int ind1_t;
typedef int ind2_t;
typedef float val_t;

#include "../../src/utils/mtx_io.h"
#include "../../src/matrix/csrc_matrix.h"
#include "../../src/matrix/dense_matrix.h"
#include "../../src/ops/aggregators.h"
#include "../common.h"

#include <torch/torch.h>

//Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DM;
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM;


struct GCN : torch::nn::Module {
    Net() {
        // TODO no liner nets for now.
//        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
//        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    }

    // Implement the Net's algorithm.
    std::vector<torch::Tensor> forward(torch::Tensor input_dense,
                          torch::Tensor offset_graph,
                          torch::Tensor columns_graph,
                          torch::Tensor value_graph) {
        return gather_forward(input_dense, offset_graph, columns_graph, value_graph);

//        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
//        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
//        x = torch::relu(fc2->forward(x));
//        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
//        return x;
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

    // Const settings
    int skip_cache_warmup = 5;

    std::string filename;
    filename = path;
    SM adj;
    std::string suffix;
    suffix = ".mtx";
    filename = path + "Adj" + suffix;
    readSM<SM>(filename, &adj);

    adj.set_all(1);

    // Adj info
    iT nrows = adj.nrows();
    iT ncols = adj.ncols();
    nT nvals = adj.nvals();

    // Init input with random numbers
    DM input_emb;
    input_emb.build(adj.ncols(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
    for (diT i = 0; i < adj.nrows(); i++) {
        for (dnT j = 0; j < emb_size; j++) {
            input_emb.vals_ptr()[i * emb_size + j] = (dvT) (rand() % 100) / 100;
        }
    }

    DM out_emb;
    out_emb.build(adj.nrows(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);

    DM out_emb2;
    out_emb2.build(adj.nrows(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);

    auto wsum_aggr = wsumAgg<val_t, val_t, ind2_t>;

    std::cout << adj.nrows() << " " << adj.ncols() << " " << adj.nvals() << std::endl;

    // Create a new Net.
    auto net = std::make_shared<GCN>();

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);
    auto output_dense = torch::zeros({nrows, dcols}, options);
    float *oden_array = output_dense.data_ptr<float>();

    int *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;

    CUDA_CHECK(cudaMalloc((void **) &dA_csrOffsets,
                          (nrows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **) &dA_columns, nvals * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **) &dA_values, nvals * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &dB, (nrows * emb_size) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA_csrOffsets, adj.offset_ptr(),
                          (nrows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_columns, adj.ids_ptr(), nvals * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_values, adj.vals_ptr(), nvals * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, input_emb.vals_ptr(), (nrows * emb_size)  * sizeof(float),
                          cudaMemcpyHostToDevice));

    auto options_cu_int = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, 1);
    torch::Tensor t_offsets = torch::from_blob(dA_csrOffsets, {nrows + 1}, options_cu_int);
    torch::Tensor t_cols = torch::from_blob(dA_columns, {nvals}, options_cu_int);

    auto options_cu_float = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, 1);
    torch::Tensor t_vals = torch::from_blob(dA_values, {nvals}, options_cu_float);
    torch::Tensor t_iden = torch::from_blob(dB, {nrows * emb_size}, options_cu_float);


    double start, end;
    std::vector<double> times_arr;
    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        size_t batch_index = 0;
        // Reset gradients.
        optimizer.zero_grad();
        // Execute the model on the input data.
        start = get_time();
        std::vector<torch::Tensor> prediction = net->forward(t_offsets, t_cols, t_vals, t_iden);
        end = get_time();

        if (epoch >= skip_cache_warmup) {
            times_arr.push_back(end - start);
        }
        // Compute a loss value to judge the prediction of our model.
//        torch::Tensor loss = torch::nll_loss(prediction, batch.target);
//        // Compute gradients of the loss w.r.t. the parameters of our model.
//        loss.backward();
//        // Update the parameters based on the calculated gradients.
//        optimizer.step();
//        // Output the loss and checkpoint every 100 batches.
//        if (++batch_index % 100 == 0) {
//            std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
//                      << " | Loss: " << loss.item<float>() << std::endl;
//            // Serialize your model periodically as a checkpoint.
//            torch::save(net, "net.pt");
//        }

    }


//    std::cout << adj.offset_ptr()[1] << " " << adj.offset_ptr()[2] << " " << adj.offset_ptr()[3] << " "
//              << adj.offset_ptr()[4] << " " << adj.offset_ptr()[5] << std::endl;
//    std::cout << adj.ids_ptr()[1] << " " << adj.ids_ptr()[2] << " " << adj.ids_ptr()[3] << " "
//              << adj.ids_ptr()[4] << " " << adj.ids_ptr()[5] << std::endl;
//    std::cout << out_emb2.vals_ptr()[0] << " " << out_emb2.vals_ptr()[0 + input_emb.ncols() * 8] << std::endl;
//    std::cout << out_emb.vals_ptr()[0] << " " << out_emb.vals_ptr()[0 + out_emb.ncols() * 8] << std::endl;

//    for (nT j = 0; j < nvals; j++) {
//        if (out_emb.vals_ptr()[j] != out_emb2.vals_ptr()[j]) {
//            std::cout << "The results don't match at: " << j << ", " << out_emb.vals_ptr()[j] << ", "
//                      << out_emb2.vals_ptr()[j] << std::endl;
//            break;
//        }
//    }
    std::cout << calc_mean(times_arr) << "," << calc_std(times_arr) << std::endl;
}