

#include <iostream>
#include <stdlib.h>

// definitions for data types
// typedef uint32_t ind1_t;
// typedef uint64_t ind2_t;
// typedef float val_t;
typedef uint32_t ind1_t;
typedef uint64_t ind2_t;
typedef float val_t;
typedef int val_int_t;
// typedef long ind1_t;
// typedef long long ind2_t;
// typedef float val_t;
// typedef int val_int_t;

#define MKL_INT ind1_t


#include "../utils/mtx_io.h"
#include "../utils/threading_utils.h"
#include "../gnn/gnn.h"
#include "../gnn/enums.h"
#include "../gnn/tiling.h"
#include "../gnn/reordering.h"
#ifdef RO_1
#include "../reorder/rabbit/rabbit_reordering.h"
#endif
#include "common.h"

//Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DMd_t;
//Dense matrix with integer values.
typedef DenseMatrix<ind1_t, ind2_t, val_int_t> DMi_t;

// Define the sparse matrix type
//#ifdef SM_1
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM_t;
//#endif
//#ifdef SM_2
//typedef COOMatrix<ind1_t,ind2_t,val_t> SM_t;
//#endif


template<class DM_t, class v_t>
void readDM(std::string filename, DM_t *mtx) {
    MtxIO<ind1_t, ind2_t, v_t> reader;
    reader.readMtx(filename);
    ind1_t nrows, ncols;
    ind2_t nvals;
    ind2_t size;
    ind1_t *col_ids, *row_ids;
    v_t *vals;
    reader.getData(nrows, ncols, nvals, size, row_ids, col_ids, vals);
    //std::cout << "Matrix: " << reader.mtx_name() << std::endl;
    //std::cout << "Nrows: " << nrows << " Ncols: " << ncols << " Nvals: " << nvals << " size: " << size << std::endl;
    mtx->build(nrows, ncols, vals, DenseMatrix<ind1_t, ind2_t, v_t>::DENSE_MTX_TYPE::RM);
}


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
#ifdef SM_1
    mtx->build(nrows, ncols, nvals, row_ids, col_ids, vals, CSRC_TYPE::CSR);
#endif
#ifdef SM_2
    mtx->build(nrows, ncols, nvals, row_ids, col_ids, vals, CSRC_TYPE::COO_RO);
#endif
}


int main(int argc, char **argv) {
    std::string path = argv[1];
    int num_layers = stoi(string(argv[2]));
    std::string filename;


    filename = path + "Adj.mtx";
    SM_t adj;
    readSM(filename, &adj);

    filename = path + "Emb.mtx";
    DMd_t input_emb;
    readDM<DMd_t, val_t>(filename, &input_emb);

    std::vector<DMd_t *> weights;
    for (int i = 1; i < num_layers; i++) {

        filename = path + "W" + std::to_string(i) + "train.mtx";
        DMd_t *weight = new DMd_t;
        readDM<DMd_t, val_t>(filename, weight);
        weights.push_back(weight);
    }

    std::vector<DMd_t *> biases;
    for (int i = 1; i < num_layers; i++) {

        filename = path + "B" + std::to_string(i) + "train.mtx";
        DMd_t *bias = new DMd_t;
        readDM<DMd_t, val_t>(filename, bias);
        biases.push_back(bias);
    }

    filename = path + "TnMsk.mtx";
    DMi_t train_masks;
    readDM<DMi_t, val_int_t>(filename, &train_masks);

    filename = path + "VlMsk.mtx";
    DMi_t valid_masks;
    readDM<DMi_t, val_int_t>(filename, &valid_masks);

    filename = path + "TsMsk.mtx";
    DMi_t test_masks;
    readDM<DMi_t, val_int_t>(filename, &test_masks);


    filename = path + "Lab.mtx";
    DMi_t labels;
    readDM<DMi_t, val_int_t>(filename, &labels);

    int rows_per_tile = stoi(string(argv[6]));
//    SM_t::itype ntiles = ceil(adj.nrows() / rows_per_tile);
    int cols_per_tile = stoi(string(argv[7]));

#ifdef LO_KK
    int dense_slice = stoi(string(argv[8]));
#endif

#ifdef RO_1
    std::unique_ptr<vint[]> perm_rabbit;
    auto nvals_var = adj.nvals();
    SM_t::itype *col_ids_var = adj.ids_ptr();
    auto vals_var = adj.vals_ptr();
    SM_t::itype *row_ids_var;
    get_row_ids<SM_t>(&adj, row_ids_var);
//    get_perm_graph<SM_t>(nvals_var, row_ids_var, col_ids_var, vals_var, perm_rabbit);
    get_perm_graph<SM_t>(&adj, perm_rabbit);
    SM_t::itype perm[adj.nrows()];
    for (SM_t::ntype p_i = 0; p_i < adj.nrows(); p_i++) {
        perm[p_i] = (SM_t::itype) perm_rabbit[p_i];
    }
    rowReorderTo(&adj, &input_emb, &train_masks, &valid_masks, &test_masks, &labels, perm);
#endif

    // TODO Change point
    double start, end;
    start = get_time();
    auto *degrees = new DMd_t;
    degrees->build(input_emb.nrows(), 1, weights[0]->type());
    SpMV_ones(&adj, degrees);
#ifdef GN_1
    auto inverse_root_operator = inverse_root<typename DMd_t::vtype>;
    UEwD(degrees, degrees, inverse_root_operator);
    auto mul_operator = mul<typename DMd_t::vtype>;
    SpVRBM(&adj, degrees, &adj, mul_operator);
    SpVCBM(&adj, degrees, &adj, mul_operator);
#endif

    int num_iters = stoi(string(argv[3]));

    SpmmVariation spmm_variation = get_variation(stoi(string(argv[4])));
    ind1_t tile_size = stoi(string(argv[5]));

    std::vector<SM_t *> tiled_adj;
    tiled_adj.push_back(&adj);

#ifdef ST_0
    std::vector<SM_t::itype> tile_offsets = static_ord_col_breakpoints<SM_t>(&adj, cols_per_tile);
    ord_col_tiling(tile_offsets, tiled_adj, 0);
#endif

    GCN_PRETILE<SM_t, DMd_t, DMi_t> gcn_pretile(tiled_adj,
                                                &input_emb,
                                                &labels,
                                                &train_masks,
                                                &valid_masks,
                                                &test_masks,
                                                weights,
                                                biases,
#ifdef GN_2
                                                degrees,
#endif
                                                num_layers);

    int i;
    for (i = 0; i < num_iters - 1; i++)
        gcn_pretile.forward_pass_split_tiled(false, rows_per_tile
#if defined(ST_1) || defined(ST_2)
                , cols_per_tile
#endif
#ifdef LO_KK
                , dense_slice
#endif
        );
    gcn_pretile.forward_pass_split_tiled(true, rows_per_tile
#if defined(ST_1) || defined(ST_2)
            , cols_per_tile
#endif
#ifdef LO_KK
            , dense_slice
#endif
    );

    end = get_time();
    std::cout << "Time of forward pass for " << num_iters << " iterations: " << end - start << std::endl;
    gcn_pretile.acc_evaluation();

    return 0;
}