
#include <iostream>
#include <stdlib.h>

// definitions for data types
typedef uint32_t ind1_t;
typedef uint64_t ind2_t;
typedef float val_t;
typedef int val_int_t;

#include "../src/utils/mtx_io.h"
#include "../src/utils/threading_utils.h"
#include "../src/gnn/gnn.h"
#ifdef RO_1
#include "../src/ops/reordering.h"
#include "../src/third_party/rabbit_reorder/rabbit_reordering.h"
#endif
#include "common.h"

//Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DMd_t;
//Dense matrix with integer values.
typedef DenseMatrix<ind1_t, ind2_t, val_int_t> DMi_t;
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM_t;

void readSM(std::string filename, SM_t *mtx) {
    MtxIO<ind1_t, ind2_t, val_t> reader;
    reader.readMtx(filename);
    ind1_t nrows, ncols;
    ind2_t nvals;
    ind2_t size;
    ind1_t *col_ids, *row_ids;
    val_t *vals;
    reader.getData(nrows, ncols, nvals, size, row_ids, col_ids, vals);
    //std::cout << "Matrix: " << reader.mtx_name() << std::endl;
    //std::cout << "Nrows: " << nrows << " Ncols: " << ncols << " Nvals: " << nvals << " size: " << size << std::endl;
    mtx->build(nrows, ncols, nvals, row_ids, col_ids, vals, CSRC_TYPE::CSR);
}


int main(int argc, char **argv) {

    std::string path = argv[1];
    int num_layers = stoi(string(argv[2]));
    std::string suffix;
    std::string filename;

    SM_t adj;
#ifdef RNPY
    suffix = ".npy";
    readSM_npy(path, &adj);
#else
    suffix = ".mtx";
    filename = path + "GAT_Adj" + suffix;
#ifdef SM_1
    readSM<SM_t>(filename, &adj);
#elif SM_2
    readSM<SM_t>(filename, &adj, CSRC_TYPE::COO_RO);
#endif
#endif


    filename = path + "GAT_Emb" + suffix;
    DMd_t input_emb;
    readDM<DMd_t>(filename, &input_emb, DMd_t::DENSE_MTX_TYPE::RM);


    std::vector<DMd_t *> weights;
    std::vector<DMd_t *> a_ls;
    std::vector<DMd_t *> a_rs;
    for (int i = 1; i < num_layers; i++) {

        filename = path + "GAT_W" + std::to_string(i) + "train" + suffix;
        DMd_t *weight = new DMd_t;
        readDM_d<DMd_t>(filename, weight, DMd_t::DENSE_MTX_TYPE::RM);
        std::cout << weight->nrows() << " " << weight->ncols() << std::endl;
        weights.push_back(weight);

        filename = path + "GAT_A_L" + std::to_string(i) + "train" + suffix;
        DMd_t *a_l = new DMd_t;
        readDM_d<DMd_t>(filename, a_l, DMd_t::DENSE_MTX_TYPE::RM);
        std::cout << a_l->nrows() << " " << a_l->ncols() << std::endl;
        a_ls.push_back(a_l);

        filename = path + "GAT_A_R" + std::to_string(i) + "train" + suffix;
        DMd_t *a_r = new DMd_t;
        readDM_d<DMd_t>(filename, a_r, DMd_t::DENSE_MTX_TYPE::RM);
        std::cout << a_r->nrows() << " " << a_r->ncols() << std::endl;
        a_rs.push_back(a_r);
    }

    filename = path + "GAT_TnMsk" + suffix;
    DMi_t train_masks;
    readDM<DMi_t>(filename, &train_masks, DMi_t::DENSE_MTX_TYPE::RM);

    filename = path + "GAT_VlMsk" + suffix;
    DMi_t valid_masks;
    readDM<DMi_t>(filename, &valid_masks, DMi_t::DENSE_MTX_TYPE::RM);

    filename = path + "GAT_TsMsk" + suffix;
    DMi_t test_masks;

    readDM<DMi_t>(filename, &test_masks, DMi_t::DENSE_MTX_TYPE::RM);

    filename = path + "GAT_Lab" + suffix;
    DMi_t labels;
    readDM<DMi_t>(filename, &labels, DMi_t::DENSE_MTX_TYPE::RM);

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

    int num_iters = stoi(string(argv[3]));
    bool fused_kernel = stoi(string(argv[4]));

    // TODO need to check if this works
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

    GAT<SM_t, DMd_t, DMi_t> my_gat(&adj,
                                   &input_emb,
                                   &labels,
                                   &train_masks,
                                   &valid_masks,
                                   &test_masks,
                                   weights,
                                   a_ls,
                                   a_rs,
                                   num_layers);

    double start, end;
    start = get_time();
    int i;
    for (i = 0; i < num_iters-1; i++)
        my_gat.forward_pass(fused_kernel,false);
    my_gat.forward_pass(fused_kernel,true);
    end = get_time();
    std::cout << "Time of forward pass for " << num_iters << " iterations: " << end - start << std::endl;
    my_gat.acc_evaluation();

    return 0;

}