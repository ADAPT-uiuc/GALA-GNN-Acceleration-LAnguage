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

#include "common.h"

#ifdef RO_1

#include "../src/ops/reordering.h"
#include "../src/third_party/rabbit_reorder/rabbit_reordering.h"

#endif

#include "../src/codegen/spade/scheduler.h"
#include "../src/codegen/spade/print_files.h"
#include "../src/codegen/spade/utils.h"
#include "../src/matrix/dense_matrix.h"

//Dense matrix with float values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DMd_t;
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM_t;


int main(int argc, char **argv) {
    // Input should be
    // 1. Input path
    // 2. Embedding size
    // 3. Number of PEs
    // 4. Colum tiling
    // 5. Row tiling
    // ?. XDense input slice // TODO need some way to incorporate this? Measure this outside of the codegen?
    // TODO Get time then multiply it by x? <- Need to do it out since this is only codegen Not needed
    // 6. *Barriered execution // TODO Not necessary
    // 7. Work division across multiple rows // TODO need to have codegen for this - Would need to have multiple
    // TODO SPADE calls for this -> Can do it like _x numer.
    // 8. Reordering
    // 9. *Prefetching // TODO not there in the current SPADE
    // 10. *Cache bypassing
    // 11. Generate empty tiles // TODO is this necessary?
    // 12. Output path



    typedef typename SM_t::itype iT;
    typedef typename SM_t::ntype nT;
    typedef typename SM_t::vtype vT;

    // Read arguments
    // Input
    std::string input_path = argv[1];
    int feature_size = stoi(string(argv[2]));
    int nPEs = stoi(string(argv[3]));

    // Configuration
    int cols_per_tile = stoi(string(argv[4]));
    int rows_per_tile = stoi(string(argv[5]));
    bool allow_barriers = stoi(string(argv[6]));
    bool work_division = stoi(string(argv[7]));
    bool reord_mtx = stoi(string(argv[8]));
    bool prefetch_data = stoi(string(argv[9]));
    bool cache_bypass = stoi(string(argv[10]));

    // Output
    bool generate_empty_tiles = stoi(string(argv[11]));
    std::string out_path = argv[12];

    // Constants
    int cache_line_size = 64;

    std::string filename;
    SM_t adj;
    ind1_t nrows, ncols;
    ind2_t nvals;
    ind2_t size;
    ind1_t *col_ids, *row_ids;
    val_t *vals;

#ifdef RNPY
    std::vector<unsigned long> shape{};
    bool fortran_order;

    filename = input_path + "Adj_src.npy";
    shape.clear();
    std::vector<uint32_t> data_adj_src;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, data_adj_src);
    nrows = (iT) data_adj_src.at(0);
    ncols = (iT) data_adj_src.at(1);

    filename = input_path + "Adj_dst.npy";
    shape.clear();
    std::vector<uint32_t> data_adj_dst;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, data_adj_dst);
    nvals = (nT) shape.at(0);
    row_ids = (iT *) aligned_alloc(64, (nvals) * sizeof(iT));
    std::copy(data_adj_src.begin() + 2, data_adj_src.end(), row_ids);
    col_ids = (iT *) aligned_alloc(64, (nvals) * sizeof(iT));
    std::copy(data_adj_dst.begin(), data_adj_dst.end(), col_ids);
    vals = (vT *) aligned_alloc(64, (nvals) * sizeof(vT));
#else
    filename = input_path;
    MtxIO<ind1_t, ind2_t, val_t> reader;
    reader.readMtx(filename);
    reader.getData(nrows, ncols, nvals, size, row_ids, col_ids, vals);
#endif
    adj.build(nrows, ncols, nvals, row_ids, col_ids, vals, CSRC_TYPE::CSR);
    adj.set_all(1);

#ifdef RO_1
    if (reord_mtx) {
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
#endif

    auto feature_mtx = new DMd_t;
#ifndef ACC_RB
    feature_mtx->build(adj.nrows(), feature_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
#endif
    print_op_config(out_path,
                    0,
                    allow_barriers,
                    prefetch_data,
                    cache_bypass);

    if (generate_empty_tiles) {
        std::vector<SM_t *> tiled_adj;
        tiled_adj.push_back(&adj);
        std::vector<SM_t::itype> tile_offsets = static_ord_col_breakpoints<SM_t>(&adj, cols_per_tile);
        ord_col_tiling_SPADE(tile_offsets, tiled_adj, 0);

        // Get tile offsets as a vector of vector for generalized print of the schedule
        std::vector<std::vector<SM_t::itype>> graph_row_tile_offsets;
        for (int col_adj_i = 0; col_adj_i < tiled_adj.size(); col_adj_i++) {
            SM_t *col_adj = tiled_adj.at(col_adj_i);
            std::vector<SM_t::itype> col_adj_row_tiles = static_ord_row_breakpoints<SM_t>(col_adj,
                                                                                          rows_per_tile);
            graph_row_tile_offsets.push_back(col_adj_row_tiles);
        }


        // Decide on the dimension for SpMM operation
        long leading_dims = feature_size;
        // Check if the number of dimensions is either divisible by the cache-line size or a multiple of it
        long bytes_in_feature = leading_dims * sizeof(DMd_t::vtype);
        assert(bytes_in_feature % cache_line_size || cache_line_size % bytes_in_feature);

        DMd_t *new_output = new DMd_t;
#ifndef ACC_RB
        new_output->build(adj.nrows(), leading_dims, feature_mtx->type());
#endif
        std::vector<SM_t::ntype> nnz_of_tiles;
        std::vector<SM_t::itype> col_of_tiles;
        get_tile_cols_nnzs<SM_t>(tiled_adj,
                                 graph_row_tile_offsets,
                                 nnz_of_tiles,
                                 col_of_tiles);

        // Print data and metadata files
        print_meta_n_data<SM_t, DMd_t>(out_path,
                                       0,
                                       SpMM,
                                       row_ids,
                                       col_ids,
                                       vals,
                                       feature_mtx,
                                       feature_mtx,
                                       new_output,
                                       leading_dims,
                                       adj.nvals(),
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
//        filter_empty_tiles(basic_schedule, nnz_of_tiles);
        // Print the schedule
        print_shedule<SM_t::ntype, int>(out_path, 0, basic_schedule);
    } else {
        // What happens with no nnz

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

        // Print data and metadata files // This handles the tiling and everything
        print_meta_n_data<SM_t, DMd_t>(out_path,
                                       0,
                                       SpMM,
                                       row_ids,
                                       col_ids,
                                       adj.vals_ptr(),
                                       feature_mtx,
                                       feature_mtx,
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