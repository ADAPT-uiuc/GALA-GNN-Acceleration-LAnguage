

#include <iostream>
#include <fstream>
#include <stdlib.h>

typedef int ind1_t;
typedef int ind2_t;
typedef float val_t;
typedef int val_int_t;

#include "../src/utils/mtx_io.h"
#include "../src/utils/threading_utils.h"

#include "../src/ops/tiling.h"
#include "../src/ops/reordering.h"

#include "common.h"
#include "../src/third_party/rabbit_reorder/rabbit_reordering.h"

#include "../cost_model/scheduler.h"
#include "../cost_model/print_files.h"
#include "../cost_model/utils.h"

//Dense matrix with float values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DMd_t;
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM_t;


int main(int argc, char **argv) {
    std::string read_path = argv[1];

    std::string out_path = argv[8];
    /// For now, assume that the number of PEs are irrelevant
    int nPEs = stoi(string(argv[9]));
    bool allow_barriers = stoi(string(argv[10]));
    long feature_size = stoi(string(argv[11]));
    bool reord_mtx = stoi(string(argv[13]));

//    int num_layers = stoi(string(argv[2]));
    std::string filename;

    filename = read_path;
    SM_t adj;
    MtxIO<ind1_t, ind2_t, val_t> reader;
    reader.readMtx(filename);
    ind1_t nrows, ncols;
    ind2_t nvals;
    ind2_t size;
    ind1_t *col_ids, *row_ids;
    val_t *vals;
    reader.getData(nrows, ncols, nvals, size, row_ids, col_ids, vals);
    adj.build(nrows, ncols, nvals, row_ids, col_ids, vals, CSRC_TYPE::CSR);
    adj.set_all(1);

    if(reord_mtx){
        std::unique_ptr<vint[]> perm_rabbit;
        auto nvals_var = adj.nvals();
        SM_t::itype *col_ids_var = adj.ids_ptr();
        auto vals_var = adj.vals_ptr();
        SM_t::itype *row_ids_var;
        get_row_ids<SM_t>(&adj, row_ids_var);
        get_perm_graph<SM_t>(nvals_var, row_ids_var, col_ids_var, vals_var, perm_rabbit);
        SM_t::itype *perm = (SM_t::itype *) aligned_alloc(64, sizeof(SM_t::itype) * nrows);
        for (SM_t::itype p_i = 0; p_i < nrows; p_i++) {
            perm[p_i] = (SM_t::itype) perm_rabbit[p_i];
        }
        rowReorderToAdj(&adj, perm);
    }

    auto feature_mtx_a = new DMd_t;
    auto feature_mtx_b = new DMd_t;

#ifndef ACC_RB
    feature_mtx_a->build(adj.nrows(), feature_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
    feature_mtx_b->build(adj.nrows(), feature_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
#endif

    int rows_per_tile = stoi(string(argv[6]));
    int cols_per_tile = stoi(string(argv[7]));
    bool generate_empty_tiles = stoi(string(argv[12]));

    if (generate_empty_tiles) {

        int cache_line_size = stoi(string(argv[12]));
        SM_t::itype cache_line_vals = cache_line_size / (int) sizeof(SM_t::vtype);


        std::vector<SM_t::itype> tile_offsets = static_ord_col_breakpoints<SM_t>(&adj, cols_per_tile);

        std::vector<SM_t *> tiled_adj;
        tiled_adj.push_back(&adj);

        ord_col_tiling_static_padding(tile_offsets, tiled_adj, 0, rows_per_tile, cache_line_vals);

        // Get tile offsets as a vector of vector for generalized print of the schedule
        std::vector<std::vector<SM_t::itype>> graph_row_tile_offsets;
        for (int col_adj_i = 0; col_adj_i < tiled_adj.size(); col_adj_i++) {
            SM_t *col_adj = tiled_adj.at(col_adj_i);
            std::vector<SM_t::itype> col_adj_row_tiles = static_ord_row_breakpoints<SM_t>(col_adj,
                                                                                          rows_per_tile);
            graph_row_tile_offsets.push_back(col_adj_row_tiles);
        }

        /// ACCODE
        // Decide on the dimension for SDDMM operation
        long leading_dims = feature_size;

        // Check if the number of dimensions is either divisible by the cache-line size or a multiple of it
        long bytes_in_feature = leading_dims * sizeof(DMd_t::vtype);
        assert(bytes_in_feature % cache_line_size || cache_line_size % bytes_in_feature);

        auto new_output = new SM_t;

        std::vector<SM_t::ntype> nnz_of_tiles;
        std::vector<SM_t::itype> col_of_tiles;
        get_tile_cols_nnzs<SM_t>(tiled_adj,
                                 graph_row_tile_offsets,
                                 nnz_of_tiles,
                                 col_of_tiles);

        SM_t::ntype total_nvals = 0;
        for (auto sub_adj: tiled_adj) {
            total_nvals += sub_adj->nvals();
        }

        // Print data and metadata files
        print_meta_n_data<SM_t, DMd_t>(out_path,
                                       0,
                                       SDDMM,
                                       row_ids,
                                       col_ids,
                                       vals,
                                       feature_mtx_a,
                                       feature_mtx_b,
                                       &adj,
                                       leading_dims,
                                       total_nvals,
                                       tiled_adj,
                                       graph_row_tile_offsets,
                                       nrows,
                                       rows_per_tile);

        // Get a schedule given the column and row tiles
        std::vector<std::vector<SM_t::ntype>> basic_schedule;
        schedule_each_tile_in_vPID<SM_t>(basic_schedule,
                                         graph_row_tile_offsets,
                                         tiled_adj,
                                         allow_barriers);

        // Print the schedule
        print_shedule<SM_t::ntype, int>(out_path, 0, basic_schedule);
    } else {
        /// ACCODE
        // Decide on the dimension for SpMM operation
        long leading_dims = feature_size;

        // Check if the number of dimensions is either divisible by the cache-line size or a multiple of it
        long bytes_in_feature = leading_dims * sizeof(DMd_t::vtype);
        assert(bytes_in_feature % cache_line_size || cache_line_size % bytes_in_feature);

        DMd_t *new_output = new DMd_t;
#ifndef ACC_RB
        new_output->build(adj.nrows(), leading_dims, feature_mtx->type());
#endif

        // Get a schedule given the column and row tiles
        std::vector<std::vector<SM_t::ntype>> basic_schedule;

        // Print data and metadata files
        print_meta_n_data<SM_t, DMd_t>(out_path,
                                       0,
                                       SpMM,
                                       row_ids,
                                       col_ids,
                                       adj.vals_ptr(),
                                       feature_mtx_a,
                                       feature_mtx_b,
                                       new_output,
                                       leading_dims,
                                       adj.nvals(),
                                       &adj,
                                       cols_per_tile,
                                       basic_schedule,
                                       nrows,
                                       rows_per_tile);

        // Print the schedule
        print_shedule<SM_t::ntype, int>(out_path, 0, basic_schedule);
    }

    return 0;
}