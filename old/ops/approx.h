#include <omp.h>
#include <math.h>
#include <chrono>
#include <unistd.h>
#include <thread>

#include "../matrix/csrc_matrix.h"
#include "../matrix//dense_matrix.h"
#include "tiling.h"

template<class SM, class DM>
void approx_range(const SM *adj, typename SM::itype *min_max, std::vector<GNOpTile<SM, DM>> &tile_infos) {
    typedef typename SM::itype iT; // Node IDs
    typedef typename SM::ntype nT; // Edge IDs
    typedef typename SM::vtype vT; // Value of node

    int cores = tile_infos.size();
    iT nrows = adj->nrows();
    auto mins = (iT *) alloca(cores * sizeof(iT));
    auto maxs = (iT *) alloca(cores * sizeof(iT));
    auto work_rows = (iT *) alloca(cores * sizeof(iT));

    auto sq_sum = (double *) alloca(cores * sizeof(double));
    auto mean = (double *) alloca(cores * sizeof(double));
    auto nm = (double *) alloca(cores * sizeof(double));

    auto entr = (double *) alloca(cores * sizeof(double));
    auto edges = (double *) alloca(cores * sizeof(double));

#pragma omp parallel for schedule(static)
    for (int c = 0; c < cores; c++) {
        iT row_start = tile_infos.at(c).srows_start;
        iT row_end = tile_infos.at(c).srows_end;
        work_rows[c] = (row_end - row_start);
        maxs[c] = 0;
        mins[c] = nrows;

        double sum_val = 0;
        double itered_over_val = 0;
        double sq_sum_val = 0;

        for (iT i = row_start; i < row_end; i++) {
            nT e_0 = adj->offset_ptr()[i];
            nT e_1 = adj->offset_ptr()[i + 1];
            nT diff_e = e_1 - e_0;
            if (diff_e < mins[c]) {
                mins[c] = diff_e;
            }
            if (diff_e > maxs[c]) {
                maxs[c] = diff_e;
            }

            sum_val += diff_e;
            sq_sum_val += diff_e * diff_e;
            itered_over_val += 1;
        }

        sq_sum[c] = sq_sum_val;
        nm[c] = itered_over_val;
        mean[c] = sum_val;
        edges[c] = sum_val;
    }

    iT min_val = nrows;
    iT max_val = 0;
    for (int c = 0; c < cores; c++) {
        if (min_val > mins[c]) {
            min_val = mins[c];
        }
        if (max_val < maxs[c]) {
            max_val = maxs[c];
        }
    }
    min_max[0] = min_val;
    min_max[1] = max_val;

    double sq_sum_vals = 0;
    double mean_sq_mul_n = 0;
    double denom_n = 0;
    double prec_edges = 0;
    iT worked_vert = 0;
    for (int c = 0; c < cores; c++) {
        sq_sum_vals += sq_sum[c];
        mean_sq_mul_n += mean[c];
        denom_n += nm[c];

        prec_edges += edges[c];

        worked_vert += work_rows[c];
    }
    mean_sq_mul_n = mean_sq_mul_n * mean_sq_mul_n / denom_n;

    double std_approx = sqrt((sq_sum_vals - mean_sq_mul_n) / denom_n);

#pragma omp parallel for schedule(static)
    for (int c = 0; c < cores; c++) {
        double sum = 0;
        iT row_start = tile_infos.at(c).srows_start;
        iT row_end = tile_infos.at(c).srows_end;
        for (iT i = row_start; i < row_end; i++) {
            nT e_0 = adj->offset_ptr()[i];
            nT e_1 = adj->offset_ptr()[i + 1];
            nT diff_e = e_1 - e_0;

            sum += ((-1) * log(diff_e / prec_edges) * diff_e);
        }
        entr[c] = sum;
    }

    double total_sum = 0;
    for (int i = 0; i < cores; i++) {
        total_sum += entr[i];
    }

    double entr_val = total_sum / prec_edges;

    std::cout << min_val << "," << max_val << ",";
    std::cout << std_approx << ",";
    std::cout << entr_val << "," << entr_val / log(worked_vert) << ",";
}

template<class SM, class DM>
void approx_reord_met(const SM *adj, std::vector<GNOpTile<SM, DM>> &tile_infos) {
    typedef typename SM::itype iT; // Node IDs
    typedef typename SM::ntype nT; // Edge IDs
    typedef typename SM::vtype vT; // Value of node

    int cores = tile_infos.size();
    iT nrows = adj->nrows();
    nT nvals = adj->nvals();
    auto src_dst_delta = (nT *) alloca(cores * sizeof(nT));
    auto closest_delta = (nT *) alloca(cores * sizeof(nT));

    auto ids_ptr = adj->ids_ptr();

    for (int c = 0; c < cores; c++) {
        src_dst_delta[c] = 0;
        closest_delta[c] = 0;
    }

#pragma omp parallel for schedule(static)
    for (int c = 0; c < cores; c++) {
        iT row_start = tile_infos.at(c).srows_start;
        iT row_end = tile_infos.at(c).srows_end;
        for (iT i = row_start; i < row_end; i++) {
            nT e_0 = adj->offset_ptr()[i];
            nT e_1 = adj->offset_ptr()[i + 1];
            if (e_1 > e_0) {
                iT u0 = ids_ptr[e_0];
                if (i > u0) {
                    src_dst_delta[c] += (i - u0);
                } else {
                    src_dst_delta[c] += (u0 - i);
                }
                if (e_1 > e_0 + 1) {
                    for (nT e = e_0 + 1; e < e_1; e++) {
                        iT u = ids_ptr[e];
                        iT u_1 = ids_ptr[e - 1];
                        if (i > u) {
                            src_dst_delta[c] += (i - u);
                        } else {
                            src_dst_delta[c] += (u - i);
                        }
                        if (u_1 > u) {
                            closest_delta[c] += (u_1 - u);
                        } else {
                            closest_delta[c] += (u - u_1);
                        }
                    }
                }
            }
        }
    }

    nT total_src_dst_delta = 0;
    nT total_closest_delta = 0;
    for (int c = 0; c < cores; c++) {
        total_src_dst_delta += src_dst_delta[c];
        total_closest_delta += closest_delta[c];
    }

    double src_dst_del = total_src_dst_delta / nvals;
    double closest_del = total_closest_delta / nvals;

    std::cout << std::sqrt(src_dst_del) << "," << std::sqrt(closest_del) << ",";
}

template<class SM>
//void gat_forward(const SM * graph, const DM* h_l0, const DM* w_l0, DM* h_l, DM* w_l){
void approx_vert_entr(const SM *graph, int prec, typename SM::itype max_nm) {
    typedef typename SM::itype iT; // Node IDs
    typedef typename SM::ntype nT; // Edge IDs
    typedef typename SM::vtype vT; // Value of node

    int rows = (int) graph->nrows();
    auto nodes_with_deg = (nT *) aligned_alloc(64, max_nm * sizeof(nT));
    for (int i = 0; i < max_nm; i++) {
        nodes_with_deg[i] = 0;
    }

    double percen = (double) prec / 100.0;
    iT iter_nm = (iT) rows * percen;

    for (iT i = 0; i < iter_nm; i++) {
        nT e_0 = graph->offset_ptr()[i];
        nT e_1 = graph->offset_ptr()[i + 1];
        nT diff_e = e_1 - e_0;

        nodes_with_deg[diff_e] += 1;
    }

    double total_sum = 0;
    auto per_ver_num = (double) iter_nm;
    for (nT i = 0; i < max_nm; i++) {
        nT vl = 1;
        if (nodes_with_deg[i] != 0) {
            vl = nodes_with_deg[i];
        }
        total_sum += ((-1) * log(vl / per_ver_num) * nodes_with_deg[i]);;
    }

    double entr_val = (total_sum / per_ver_num);
    std::cout << entr_val;
}

// TODO Edge entropy and degree entropy?
