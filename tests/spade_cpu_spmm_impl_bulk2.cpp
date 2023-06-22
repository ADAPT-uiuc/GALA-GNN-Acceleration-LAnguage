//
// Created by damitha on 5/10/22.
//


#include <iostream>
#include <stdlib.h>

typedef uint32_t ind1_t;
typedef uint64_t ind2_t;
typedef float val_t;

#include "../src/utils/mtx_io.h"
#include "../src/utils/threading_utils.h"
#include "../src/matrix/csrc_matrix.h"
#include "../src/matrix/dense_matrix.h"
#include "../src/ops/aggregators.h"
#include "../src/ops/sparse_matrix_ops.h"
#include "../src/ops/tiling.h"
#include "common.h"

#ifdef RO_1

#include "../src/ops/reordering.h"
#include "../src/third_party/rabbit_reorder/rabbit_reordering.h"

#endif

//Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DM;
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM;

int main(int argc, char **argv) {
    // Input should be
    // 1. Input path for data

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    std::string path = argv[1];
    int exec_feat_group = stoi(string(argv[2]));

    // Test parameters settings
    // Embedding sizes
    diT emb_arr[2] = {32, 128};
    if (exec_feat_group == 1) {
        emb_arr[0] = 512;
        emb_arr[1] = 2048;
    }

    // Test opt params
    // Col tile sizes
    iT col_arr[8] = {512, 2048, 8192, 32768, 131072, 524228, 2097152};
    // Row tile sizes
    iT row_arr[7] = {1, 4, 16, 64, 256, 1024, 4096};
    // Slice sizes
    diT slice_arr[4] = {32, 128, 512};
    // Do reordering as well
    bool reord_arr[2] = {false, true};
    // TODO Barriers
    // TODO Prefetch
    // TODO Work division

    // Iteration configurations
    int skip_cache_warmup = 2;
    int max_num_iters = 100;
    int threshold_s = 1;
    // Semiring
    auto wsum_aggr = wsumAgg<val_t, val_t, ind2_t>;
    // Timing init
    double start, end, total;
    std::vector<double> times_arr;
    std::tuple<double, double> out_times;

    std::string filename;
    filename = path;
    SM adj;
    std::string suffix;
#ifdef RNPY
    suffix = ".npy";
    readSM_npy32(path, &adj);
#else
    suffix = ".mtx";
    filename = path + "Adj" + suffix;
    readSM<SM_t>(filename, &adj);
#endif
    adj.set_all(1);

    // Adj info
    iT nrows = adj.nrows();
    iT ncols = adj.ncols();
    nT nvals = adj.nvals();

    // One time reordering
    SM adj_reord;
    adj_reord.clone_mtx(nrows,
                        ncols,
                        nvals,
                        adj.ids_ptr(),
                        adj.vals_ptr(),
                        adj.offset_ptr(),
                        adj.type());
    std::unique_ptr<vint[]> perm_rabbit;
    auto nvals_var = adj_reord.nvals();
    auto nrows_var = adj_reord.nrows();
    iT *col_ids_var = adj_reord.ids_ptr();
    auto vals_var = adj_reord.vals_ptr();
    iT *row_ids_var;
    get_row_ids<SM>(&adj_reord, row_ids_var);
    get_perm_graph<SM>(nrows_var, nvals_var, row_ids_var, col_ids_var, vals_var, perm_rabbit);
    iT *perm = (iT *) aligned_alloc(64, sizeof(iT) * nrows);
    for (iT p_i = 0; p_i < nrows; p_i++) {
        perm[p_i] = (iT) perm_rabbit[p_i];
    }
    rowReorderToAdj(&adj_reord, perm);

    // One time map to store the column tile segments
    std::map<iT, std::vector<SM *>> tile_adj_map;
    std::map<iT, std::vector<SM *>> reord_tile_adj_map;
    for (auto cols_per_tile: col_arr) {
        if (cols_per_tile > ncols) {
            break;
        }

        // Segment the input matrix
        std::vector<iT> tile_offsets;
        std::vector<iT> reord_tile_offsets;


        // For each col tile size, reinit tiling set
        std::vector<SM *> tiled_adj;
        tiled_adj.push_back(&adj);
        tile_offsets = static_ord_col_breakpoints<SM>(&adj, cols_per_tile);
        ord_col_tiling(tile_offsets, tiled_adj, 0);
        tile_adj_map[cols_per_tile] = tiled_adj;

        std::vector<SM *> reord_tiled_adj;
        reord_tiled_adj.push_back(&adj_reord);
        reord_tile_offsets = static_ord_col_breakpoints<SM>(&adj_reord, cols_per_tile);
        ord_col_tiling(reord_tile_offsets, reord_tiled_adj, 0);
        reord_tile_adj_map[cols_per_tile] = reord_tiled_adj;

        tile_offsets.clear();
        reord_tile_offsets.clear();
    }

    for (auto emb_size: emb_arr) {
        // TODO init slices
        // Init input with random numbers
        DM input_emb;
        input_emb.build(adj.nrows(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
        for (diT i = 0; i < adj.nrows(); i++) {
            for (dnT j = 0; j < emb_size; j++) {
                input_emb.vals_ptr()[i * emb_size + j] = (dvT) (rand() % 100) / 100;
            }
        }
        DM out_emb;
        out_emb.build(adj.nrows(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);

        // Create slices for each input and output
        std::map<diT, std::vector<DM *>> slice_inp_map;
        std::map<diT, std::vector<DM *>> slice_out_map;
        for (auto slice_size: slice_arr) {
            if (slice_size >= emb_size) {
                break;
            }
            // Init sliced inputs
            std::vector<DM *> sliced_inp;
            slice_tiling(slice_size, &input_emb, sliced_inp);
            slice_inp_map[slice_size] = sliced_inp;
            // Init sliced output
            std::vector<DM *> sliced_out;
            slice_tiling(slice_size, &out_emb, sliced_out);
            slice_out_map[slice_size] = sliced_out;
        }

        // Warmup cache
        for (int i = 0; i < 5; i++) {
            gSpMM(&adj, &input_emb, &out_emb, wsum_aggr);
        }

        for (auto reord_mtx: reord_arr) {
            for (auto cols_per_tile: col_arr) {
                if (cols_per_tile > ncols) {
                    break;
                }

                // For each col tile size, reinit tiling set
                std::vector<SM *> tiled_adj;
                if (reord_mtx) {
                    tiled_adj = reord_tile_adj_map[cols_per_tile];
                } else {
                    tiled_adj = tile_adj_map[cols_per_tile];
                }

                // Row tile
                for (auto rows_per_tile: row_arr) {

                    // Unsliced execution
                    total = 0;
                    times_arr.clear();
                    for (int i = 0; i < max_num_iters + skip_cache_warmup; i++) {
                        out_emb.set_all(0);

                        start = get_time();
                        trans_jj_iip_i_j_kv(tiled_adj,
                                            &input_emb,
                                            &out_emb,
                                            rows_per_tile,
                                            wsum_aggr);

                        end = get_time();

                        if (i >= skip_cache_warmup) {
                            times_arr.push_back(end - start);
                            total += (end - start);

                            if (total > threshold_s) {
                                break;
                            }
                        }
                    }
                    out_times = calc_mean_std(times_arr);
                    std::cout << emb_size << "," << cols_per_tile << "," << rows_per_tile << ","
                              << "no_slice,no_barr,no_wdiv,0,no_pref,"
                              << std::get<0>(out_times) << ","
                              << std::get<1>(out_times) << std::endl;

                    if (!reord_mtx) {
                        // Sliced execution
                        for (auto slice_size: slice_arr) {
                            if (slice_size >= emb_size) {
                                break;
                            }

                            total = 0;
                            times_arr.clear();
                            for (int i = 0; i < max_num_iters + skip_cache_warmup; i++) {
                                std::vector<DM *> sliced_inp = slice_inp_map[slice_size];
                                std::vector<DM *> sliced_out = slice_out_map[slice_size];
                                for (auto slice_out: sliced_out){
                                    slice_out->set_all(0);
                                }

                                start = get_time();
                                slice_kk_jj_iip_i_j_kv(tiled_adj,
                                                       sliced_inp,
                                                       sliced_out,
                                                       rows_per_tile,
                                                       wsum_aggr);
                                end = get_time();

                                if (i >= skip_cache_warmup) {
                                    times_arr.push_back(end - start);
                                    total += (end - start);

                                    if (total > threshold_s) {
                                        break;
                                    }
                                }
                            }
                            out_times = calc_mean_std(times_arr);
                            std::cout << emb_size << "," << cols_per_tile << "," << rows_per_tile << ","
                                      << slice_size << ",no_barr,no_wdiv,0,no_pref,"
                                      << std::get<0>(out_times) << ","
                                      << std::get<1>(out_times) << std::endl;
                        }
                    }
                }
            }

            // Reinit tiled adj with the original untiled matrix
            std::vector<SM *> tiled_adj;
            if (reord_mtx) {
                tiled_adj.push_back(&adj_reord);
            } else {
                tiled_adj.push_back(&adj);
            }

            // Unsliced version
            total = 0;
            times_arr.clear();
            for (int i = 0; i < max_num_iters + skip_cache_warmup; i++) {
                out_emb.set_all(0);

                start = get_time();
                trans_jj_iip_i_j_kv(tiled_adj,
                                    &input_emb,
                                    &out_emb,
                                    1,
                                    wsum_aggr);
                end = get_time();

                if (i >= skip_cache_warmup) {
                    times_arr.push_back(end - start);
                    total += (end - start);

                    if (total > threshold_s) {
                        break;
                    }
                }
            }
            out_times = calc_mean_std(times_arr);
            std::cout << emb_size << "," << ncols << "," << 1 << ","
                      << "no_slice,no_barr,no_wdiv,0,no_pref,"
                      << std::get<0>(out_times) << ","
                      << std::get<1>(out_times) << std::endl;

            // Sliced execution
            for (auto slice_size: slice_arr) {
                if (slice_size >= emb_size) {
                    break;
                }
                total = 0;
                times_arr.clear();
                for (int i = 0; i < max_num_iters + skip_cache_warmup; i++) {
                    std::vector<DM *> sliced_inp = slice_inp_map[slice_size];
                    std::vector<DM *> sliced_out = slice_out_map[slice_size];
                    for (auto slice_out: sliced_out){
                        slice_out->set_all(0);
                    }

                    start = get_time();
                    slice_kk_jj_iip_i_j_kv(tiled_adj,
                                           sliced_inp,
                                           sliced_out,
                                           1,
                                           wsum_aggr);
                    end = get_time();

                    if (i >= skip_cache_warmup) {
                        times_arr.push_back(end - start);
                        total += (end - start);

                        if (total > threshold_s) {
                            break;
                        }
                    }
                }
                out_times = calc_mean_std(times_arr);
                std::cout << emb_size << "," << ncols << "," << 1 << ","
                          << slice_size << ",no_barr,no_wdiv,0,no_pref,"
                          << std::get<0>(out_times) << ","
                          << std::get<1>(out_times) << std::endl;
            }
        }

        // Cleanup
//        for (auto slice_size: slice_arr) {
//            if (slice_size >= emb_size) {
//                break;
//            }
//
//            std::vector<DM *> sliced_inp = slice_inp_map[slice_size];
//            for (auto slice_inp: sliced_inp){
//                slice_inp->clear();
//            }
//            std::vector<DM *> sliced_out = slice_out_map[slice_size];
//            for (auto slice_out: sliced_out){
//                slice_out->clear();
//            }
//
//            sliced_inp.clear();
//            sliced_out.clear();
//        }
        slice_inp_map.clear();
        slice_out_map.clear();
        input_emb.clear();
        out_emb.clear();
    }
    return 0;
}