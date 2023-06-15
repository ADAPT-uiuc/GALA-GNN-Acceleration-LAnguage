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

    // Test parameters settings
    // Embedding sizes
    diT emb_arr[4] = {32, 128, 512, 2048};

    // Test opt params
    // Col tile sizes
    iT col_arr[8] = {512, 2048, 8192, 32768, 131072, 524228, 2097152};
    // Row tile sizes
    iT row_arr[7] = {1, 4, 16, 64, 256, 1024, 4096};
    // Slice sizes
    diT slice_arr[4] = {32, 128, 512, 2048};
    // Do reordering as well
    bool reord_arr[2] = {false, true};
    // Loop orderings
    int loop_ord_arr[3] = {2, 3, 5};
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
    int i;
    std::vector<double> times_arr;

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

    for (auto emb_size: emb_arr) {
        // Init input with random numbers
        DM input_emb;
        input_emb.build(adj.nrows(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
        for (diT i = 0; i < adj.nrows(); i++) {
            for (dnT j = 0; j < emb_size; j++) {
                input_emb.vals_ptr()[i * emb_size + j] = (dvT) (rand() % 100) / 100;
            }
        }
        input_emb.set_all(1);

        DM out_emb;
        out_emb.build(adj.nrows(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);

        for (auto reord_mtx: reord_arr) {
            std::vector<SM *> tiled_adj;

            if (reord_mtx) {
                tiled_adj.push_back(&adj_reord);
            } else {
                tiled_adj.push_back(&adj);
            }
            for (auto cols_per_tile: col_arr) {
                if (cols_per_tile > ncols){
                    break;
                }

                // Segment the input matrix
                std::vector<iT> tile_offsets;
                if (reord_mtx) {
                    tile_offsets = static_ord_col_breakpoints<SM>(&adj_reord, cols_per_tile);
                } else {
                    tile_offsets = static_ord_col_breakpoints<SM>(&adj, cols_per_tile);
                }
                ord_col_tiling(tile_offsets, tiled_adj, 0);

                for (auto rows_per_tile: row_arr) {
                    if (!reord_mtx) {
                        for (int loop_ord: loop_ord_arr) {
                            if (loop_ord == 2) {
                                total = 0;
                                times_arr.clear();
                                for (i = 0; i < max_num_iters + skip_cache_warmup; i++) {
                                    out_emb.set_all(0);

                                    start = get_time();
                                    tile_jj_ii_i_j_kv(tiled_adj,
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
                                std::tuple<int, int> out_times = calc_mean_std(times_arr);
                                std::cout << emb_size << "," << cols_per_tile << "," << rows_per_tile << "," << loop_ord
                                          << ",no_slice,no_barr,no_wdiv,0,no_pref," << std::get<0>(out_times) << ","
                                          << std::get<1>(out_times);
                            } else {
                                for (auto slice_size: slice_arr) {
                                    if (loop_ord == 3) {
                                        total = 0;
                                        times_arr.clear();
                                        for (i = 0; i < max_num_iters + skip_cache_warmup; i++) {
                                            out_emb.set_all(0);

                                            start = get_time();
                                            tile_kk_jj_iip_i_j_kv(tiled_adj,
                                                                  &input_emb,
                                                                  &out_emb,
                                                                  rows_per_tile,
                                                                  wsum_aggr,
                                                                  slice_size);
                                            end = get_time();

                                            if (i >= skip_cache_warmup) {
                                                times_arr.push_back(end - start);
                                                total += (end - start);

                                                if (total > threshold_s) {
                                                    break;
                                                }
                                            }
                                        }
                                        std::tuple<int, int> out_times = calc_mean_std(times_arr);
                                        std::cout << emb_size << "," << cols_per_tile << "," << rows_per_tile << ","
                                                  << loop_ord
                                                  << "," << slice_size << ",no_barr,no_wdiv,0,no_pref,"
                                                  << std::get<0>(out_times) << ","
                                                  << std::get<1>(out_times);
                                    } else if (loop_ord == 5) {
                                        total = 0;
                                        times_arr.clear();
                                        for (i = 0; i < max_num_iters + skip_cache_warmup; i++) {
                                            out_emb.set_all(0);

                                            start = get_time();
                                            tile_jj_kk_iip_i_j_kv(tiled_adj,
                                                                  &input_emb,
                                                                  &out_emb,
                                                                  rows_per_tile,
                                                                  wsum_aggr,
                                                                  slice_size);
                                            end = get_time();

                                            if (i >= skip_cache_warmup) {
                                                times_arr.push_back(end - start);
                                                total += (end - start);

                                                if (total > threshold_s) {
                                                    break;
                                                }
                                            }
                                        }
                                        std::tuple<int, int> out_times = calc_mean_std(times_arr);
                                        std::cout << emb_size << "," << cols_per_tile << "," << rows_per_tile << ","
                                                  << loop_ord
                                                  << "," << slice_size << ",no_barr,no_wdiv,0,no_pref,"
                                                  << std::get<0>(out_times) << ","
                                                  << std::get<1>(out_times);

                                    }
                                }
                            }
                        }
                    } else {
                        total = 0;
                        times_arr.clear();

                        int loop_ord = 2;
                        for (i = 0; i < max_num_iters + skip_cache_warmup; i++) {
                            out_emb.set_all(0);

                            start = get_time();
                            tile_jj_ii_i_j_kv(tiled_adj,
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
                        std::tuple<int, int> out_times = calc_mean_std(times_arr);
                        std::cout << emb_size << "," << cols_per_tile << "," << rows_per_tile << "," << loop_ord
                                  << ",no_slice,no_barr,no_wdiv,0,no_pref," << std::get<0>(out_times) << ","
                                  << std::get<1>(out_times);
                    }
                }
            }

            // TODO add reordering and slicing
            for (int loop_ord: loop_ord_arr) {
                if (loop_ord == 2) {
                    total = 0;
                    times_arr.clear();
                    for (i = 0; i < max_num_iters + skip_cache_warmup; i++) {
                        out_emb.set_all(0);

                        start = get_time();
                        tile_jj_ii_i_j_kv(tiled_adj,
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
                    std::tuple<int, int> out_times = calc_mean_std(times_arr);
                    std::cout << emb_size << "," << ncols << "," << 1 << "," << loop_ord
                              << ",no_slice,no_barr,no_wdiv,0,no_pref," << std::get<0>(out_times) << ","
                              << std::get<1>(out_times);
                } else {
                    for (auto slice_size: slice_arr) {
                        if (loop_ord == 3) {
                            total = 0;
                            times_arr.clear();
                            for (i = 0; i < max_num_iters + skip_cache_warmup; i++) {
                                out_emb.set_all(0);

                                start = get_time();
                                tile_kk_jj_iip_i_j_kv(tiled_adj,
                                                      &input_emb,
                                                      &out_emb,
                                                      1,
                                                      wsum_aggr,
                                                      slice_size);
                                end = get_time();

                                if (i >= skip_cache_warmup) {
                                    times_arr.push_back(end - start);
                                    total += (end - start);

                                    if (total > threshold_s) {
                                        break;
                                    }
                                }
                            }
                            std::tuple<int, int> out_times = calc_mean_std(times_arr);
                            std::cout << emb_size << "," << ncols << "," << 1 << ","
                                      << loop_ord
                                      << "," << slice_size << ",no_barr,no_wdiv,0,no_pref,"
                                      << std::get<0>(out_times) << ","
                                      << std::get<1>(out_times);
                        } else if (loop_ord == 5) {
                            total = 0;
                            times_arr.clear();
                            for (i = 0; i < max_num_iters + skip_cache_warmup; i++) {
                                out_emb.set_all(0);

                                start = get_time();
                                tile_jj_kk_iip_i_j_kv(tiled_adj,
                                                      &input_emb,
                                                      &out_emb,
                                                      1,
                                                      wsum_aggr,
                                                      slice_size);
                                end = get_time();

                                if (i >= skip_cache_warmup) {
                                    times_arr.push_back(end - start);
                                    total += (end - start);

                                    if (total > threshold_s) {
                                        break;
                                    }
                                }
                            }
                            std::tuple<int, int> out_times = calc_mean_std(times_arr);
                            std::cout << emb_size << "," << ncols << "," << 1 << ","
                                      << loop_ord
                                      << "," << slice_size << ",no_barr,no_wdiv,0,no_pref,"
                                      << std::get<0>(out_times) << ","
                                      << std::get<1>(out_times);
                        }
                    }
                }
            }
        }
    }
    return 0;
}