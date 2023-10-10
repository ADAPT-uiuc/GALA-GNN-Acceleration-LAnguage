

#include <iostream>
#include <stdlib.h>

// definitions for data types
// typedef uint32_t ind1_t;
// typedef uint64_t ind2_t;
// typedef float val_t;

// #ifdef TMKL
// #include "mkl.h"
// #include "mkl_types.h"
// #endif

// #ifdef TMKL
// typedef MKL_INT ind1_t;
// #else
// typedef uint32_t ind1_t;
// #endif

// #ifdef TMKL
// typedef MKL_INT ind2_t;
// #else
// typedef uint64_t ind2_t;
// #endif

#ifdef TMKL
typedef long long int ind1_t;
#else
typedef uint32_t ind1_t;
#endif

#ifdef TMKL
typedef long long int ind2_t;
#else
typedef uint64_t ind2_t;
#endif
typedef float val_t;
typedef int val_int_t;
// typedef long ind1_t;
// typedef long long ind2_t;
// typedef float val_t;
// typedef int val_int_t;

#ifdef TMKL
#define MKL_INT ind1_t
#endif

#include "../src/utils/mtx_io.h"
#include "../src/utils/threading_utils.h"
#include "../src/gnn/gnn.h"
#include "../src/ops/enums.h"
#include "common.h"

#ifdef RO_1
#include "../src/ops/reordering.h"
#include "../src/third_party/rabbit_reorder/rabbit_reordering.h"
#endif

//Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DMd_t;
//Dense matrix with integer values.
typedef DenseMatrix<ind1_t, ind2_t, val_int_t> DMi_t;
#if defined(SM_1) || defined(SM_3)
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM_t;
#elif SM_2
typedef COOMatrix<ind1_t,ind2_t,val_t> SM_t;
#endif

int main(int argc, char **argv) {
    std::string path = argv[1];
    int num_layers = stoi(string(argv[2]));
    std::string filename;
    std::string suffix;
//    std::string prefix = "dyn";
    std::string prefix = "pre";

    SM_t adj;
    DMd_t input_emb;
    std::vector<DMd_t *> weights;
    std::vector<DMd_t *> biases;
    DMi_t train_masks;
    DMi_t valid_masks;
    DMi_t test_masks;
    DMi_t labels;

//    readSM_npy("/home/damitha/dsl-sacc/data_pyarr/", &adj);
#ifdef RNPY
    suffix = ".npy";
    readSM_npy32(path, &adj);
#else
    suffix = ".mtx";
    filename = path + "Adj" + suffix;
#ifdef SM_1
    readSM<SM_t>(filename, &adj);
#elif SM_2
    readSM<SM_t>(filename, &adj, CSRC_TYPE::COO_RO);
#endif
#endif
//    for (ind1_t i = 0; i < adj.nrows(); i++){
//        for(ind2_t j = adj.offset_ptr()[i]; j < adj.offset_ptr()[i+1]; j++){
//            std::cout << i+1 << " " << adj.ids_ptr()[j]+1 << " " << adj.vals_ptr()[j] << std::endl;
//        }
//    }

    filename = path + "Emb" + suffix;
    readDM<DMd_t>(filename, &input_emb, DMd_t::DENSE_MTX_TYPE::RM);
    for (int i = 1; i < num_layers; i++) {
#ifdef IMAST
        filename = path + prefix + "W" + std::to_string(i) + suffix;
#else
        filename = path + prefix + "W" + std::to_string(i) + "train" + suffix;
#endif
        DMd_t *weight = new DMd_t;
        readDM<DMd_t>(filename, weight, DMd_t::DENSE_MTX_TYPE::RM);
        weights.push_back(weight);
    }
    for (int i = 1; i < num_layers; i++) {
#ifdef IMAST
        filename = path + prefix + "B" + std::to_string(i) + suffix;
#else
        filename = path + prefix + "B" + std::to_string(i) + "train" + suffix;
#endif
        DMd_t *bias = new DMd_t;
        readDV<DMd_t>(filename, bias, DMd_t::DENSE_MTX_TYPE::RM);
        biases.push_back(bias);
    }
    filename = path + "TnMsk" + suffix;
    readDM<DMi_t>(filename, &train_masks, DMi_t::DENSE_MTX_TYPE::RM);

    filename = path + "VlMsk" + suffix;
    readDM<DMi_t>(filename, &valid_masks, DMi_t::DENSE_MTX_TYPE::RM);

    filename = path + "TsMsk" + suffix;
    readDM<DMi_t>(filename, &test_masks, DMi_t::DENSE_MTX_TYPE::RM);

    filename = path + "Lab" + suffix;
    readDM<DMi_t>(filename, &labels, DMi_t::DENSE_MTX_TYPE::RM);

    double start, end;
    start = get_time();

#ifdef RO_1
    std::unique_ptr<vint[]> perm_rabbit;
    auto nvals_var = adj.nvals();
    SM_t::itype *col_ids_var = adj.ids_ptr();
    auto vals_var = adj.vals_ptr();
    SM_t::itype *row_ids_var;
    get_row_ids<SM_t>(&adj, row_ids_var);
    get_perm_graph<SM_t>(nvals_var, row_ids_var, col_ids_var, vals_var, perm_rabbit);
    SM_t::itype perm[adj.nrows()];
    for (SM_t::ntype p_i = 0; p_i < adj.nrows(); p_i++) {
        perm[p_i] = (SM_t::itype) perm_rabbit[p_i];
    }
    rowReorderTo(&adj, &input_emb, &train_masks, &valid_masks, &test_masks, &labels, perm);
#endif

//    std::cout << "Works 1.1" << std::endl;
//
//    std::cout << adj.row_ids_ptr()[0] << " " << adj.col_ids_ptr()[0] << std::endl;
//    std::cout << adj.row_ids_ptr()[1] << " " << adj.col_ids_ptr()[1] << std::endl;
//    std::cout << adj.row_ids_ptr()[2] << " " << adj.col_ids_ptr()[2] << std::endl;
//
//    std::cout << "Works 4" << std::endl;

    // TODO better if it was moved into a gnn
#ifdef GN_1
    DMd_t *degrees = new DMd_t;
    degrees->build(input_emb.nrows(), 1, weights[0]->type());
    degrees->set_all(0);
    SpMV_ones(&adj, degrees);
    auto inverse_root_operator = inverse_root<typename DMd_t::vtype>;
    UEwD(degrees, degrees, inverse_root_operator);
    auto mul_operator = mul<typename DMd_t::vtype>;
    SpVRBM(&adj, degrees, &adj, mul_operator);
    SpVCBM(&adj, degrees, &adj, mul_operator);
#endif
//    std::cout << "Works 2" << std::endl;


    int num_iters = stoi(string(argv[3]));

    SpmmVariation spmm_variation = get_variation(stoi(string(argv[4])));
    ind1_t tile_size = stoi(string(argv[5]));

#ifdef TMKL
    auto off_ptr = adj.offset_ptr();
    auto off_ptr2 = &adj.offset_ptr()[1];

    auto A = new sparse_matrix_t;
    struct matrix_descr descrA;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

    mkl_sparse_s_create_csr(A,
                            SPARSE_INDEX_BASE_ZERO,
                            adj.nrows(),
                            adj.ncols(),
                            off_ptr,
                            off_ptr2,
                            adj.ids_ptr(),
                            adj.vals_ptr());

    mkl_sparse_set_memory_hint(*A, SPARSE_MEMORY_AGGRESSIVE);
    mkl_sparse_set_mm_hint(*A, SPARSE_OPERATION_NON_TRANSPOSE, descrA, SPARSE_LAYOUT_ROW_MAJOR, input_emb.ncols(),
                           num_iters);
    mkl_sparse_optimize(*A);
#endif

    GCN<SM_t, DMd_t, DMi_t> gcn(&adj,
                                &input_emb,
                                &labels,
                                &train_masks,
                                &valid_masks,
                                &test_masks,
                                weights,
                                biases,
                                num_layers);

//    std::cout << "Works 3" << std::endl;

    int i;
#ifdef TMKL
    std::cout << "MKL runs" << std::endl;
    for (i = 0; i < num_iters - 1; i++)
        gcn.forward_pass_mkl(spmm_variation,
                             tile_size,
                             false,
                             A,
                             descrA);
    gcn.forward_pass_mkl(spmm_variation,
                         tile_size,
                         true,
                         A,
                         descrA);
#else
    for (i = 0; i < num_iters - 1; i++)
        gcn.forward_pass(spmm_variation,
                         tile_size,
                         false);
    gcn.forward_pass(spmm_variation,
                     tile_size,
                     true);
#endif
    end = get_time();
    std::cout << "Time of forward pass for " << num_iters << " iterations: " << end - start << std::endl;
    gcn.acc_evaluation();

    return 0;
}