#include <omp.h>

#include "../matrix/csrc_matrix.h"
#include "../matrix/dense_matrix.h"
#include "../utils/threading_utils.h"
#include "../ops/sparse_matrix_ops.h"
#include "../ops/dense_matrix_ops.h"
#include "../ops/aggregators.h"
#include "../ops/uelw_ops.h"
#include "../ops/belw_ops.h"


template<class SM, class DM>
void gat_forward_layer(SM *adj,
                       DM *h_in,
                       DM *w,
                       DM *a_l,
                       DM *a_r,
                       DM *h_out,
                       DM *wh1,
                       DM *wh2,
                       bool recomp,
                       bool refresh_and_print,
                       int layer,
                       double **&individual_layer_times) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double upd_time = 0.0;
    static double wh1_time = 0.0;
    static double wh2_time = 0.0;
    static double edge_time = 0.0;
    static double spmm_time = 0.0;
    static double elw_time = 0.0;

    DM h_temp;
    DM h_temp2;
#ifdef A_ALLOC
    h_temp.build(h_out->nrows(), h_out->ncols(), h_out->type(), 0);
#else
    h_temp.build(h_out->nrows(), h_out->ncols(), h_out->type());
#endif
    h_temp.set_all(0); // These may be redundant
    h_out->set_all(0);

    double t0 = get_time();

//    std::cout << h_in->vals_ptr()[0] << " " << h_in->vals_ptr()[1] << std::endl;
//    std::cout << w->vals_ptr()[0] << " " << w->vals_ptr()[1] << " " << w->vals_ptr()[2] << std::endl;
//    std::cout << w->vals_ptr()[0 + w->ncols()] << " " << w->vals_ptr()[1 + w->ncols()] << " " << w->vals_ptr()[2 + w->ncols()] << std::endl;

    MM_mkl(h_in, w, &h_temp);


//    std::cout << h_temp.vals_ptr()[0] << " " << h_temp.vals_ptr()[1] << std::endl;

    double t1 = get_time();

    /// Calculate WH1
    MM_mkl(&h_temp, a_r, wh1);

    double t2 = get_time();

    /// Calculate WH2
    MM_mkl(&h_temp, a_l, wh2);

    double t3 = get_time();

    /// Edge processing
    auto sum_operator = sum<vT>;
    gSDDVV(adj, wh1, wh2, adj, sum_operator);


    //Apply Edge = Unary ElementWise Operation on Sparse Matrix.
    auto explrelu_operator = explrelu<vT>;
    UEwS(adj, adj, explrelu_operator);


    //SpMV_ones (special case- I don't express it with generalized for extra efficiency).
    DM sums;
    sums.build(h_in->nrows(), 1, h_in->type());
    SpMV_ones(adj, &sums);


    auto adj_vals_ptr = adj->vals_ptr();
    auto sums_vals_ptr = sums.vals_ptr();
    auto adj_offset = adj->offset_ptr();
    auto adj_nrows = adj->nrows();
    //gSDDVV (maybe not best formulation)
#pragma omp parallel for schedule(dynamic, 1)
    for (iT v = 0; v < adj_nrows; v++) {
        auto e_start = adj_offset[v];
        auto e_end = adj_offset[v + 1];
        for (nT e = e_start; e < e_end; e++) {
            adj_vals_ptr[e] = adj_vals_ptr[e] / sums_vals_ptr[(nT) v];
        }
    }
//    if (fused_edge_proc) {
//        // TODO check if this is faster or not
//        fusedGatEdgeKernel<DM, SM>(adj, wh1, wh2, adj);
//    } else {
//        //gSDDVV
//        auto sum_operator = sum<vT>;
//        gSDDVV(adj, wh1, wh2, adj, sum_operator);
//
//
//        //Apply Edge = Unary ElementWise Operation on Sparse Matrix.
//        auto explrelu_operator = explrelu<vT>;
//        UEwS(adj, adj, explrelu_operator);
//
//
//        //SpMV_ones (special case- I don't express it with generalized for extra efficiency).
//        DM sums;
//        sums.build(h_in->nrows(), 1, h_in->type());
//        SpMV_ones(adj, &sums);
//
//
//        //gSDDVV (maybe not best formulation)
//#pragma omp parallel for
//        for (iT v = 0; v < adj->nrows(); v++) {
//            for (nT e = adj->offset_ptr()[v]; e < adj->offset_ptr()[v + 1]; e++) {
//                adj->vals_ptr()[e] = adj->vals_ptr()[e] / sums.vals_ptr()[(nT) v];
//            }
//        }
//    }
    double t4 = get_time();


    /// Node processing
    //Defusing the gSpMM kernel and the Dense element wise Relu yielded acceleration.

    auto wsum_aggr = wsumAgg<dvT, vT, dnT>;

    // TODO need to make this a parameter(iszed)
    if (h_in->ncols() < h_out->ncols() && recomp) {
#ifdef A_ALLOC
        h_temp2.build(h_in->nrows(), h_in->ncols(), h_in->type(), 0);
#else
        h_temp2.build(h_in->nrows(), h_in->ncols(), h_in->type());
#endif
        h_temp2.set_all(0);
        gSpMM_set<DM, SM>(adj, h_in, &h_temp2, wsum_aggr);
        MM_mkl(&h_temp2, w, h_out);
    } else {
        gSpMM_set<DM, SM>(adj, &h_temp, h_out, wsum_aggr);
    }

    double t5 = get_time();

    //Unary element-wise operation on Dense Matrix: ReLU
    auto relu_operator = relu<vT>;
    UEwD<DM>(h_out, h_out, relu_operator);

    double t6 = get_time();
    upd_time += t1 - t0;
    wh1_time += t2 - t1;
    wh2_time += t3 - t2;
    edge_time += t4 - t3;
    spmm_time += t5 - t4;
    elw_time += t6 - t5;

    individual_layer_times[layer][0] += t5 - t4;
    individual_layer_times[layer][1] += t1 - t0;
    individual_layer_times[layer][3] += ((t2 - t1) + (t3 - t2));
    individual_layer_times[layer][4] += t4 - t3;

    h_temp.clear();
    h_temp2.clear();

    if (refresh_and_print) {
        std::cout << "Dense Update took: " << upd_time << std::endl;
        std::cout << "Dense WH1 calculation took: " << wh1_time << std::endl;
        std::cout << "Dense WH2 calculation took: " << wh2_time << std::endl;
        std::cout << "Edge processing took: " << edge_time << std::endl;
        std::cout << "SpMM took: " << spmm_time << std::endl;
        std::cout << "Element wise ReLU on Dense took: " << elw_time << std::endl;
        upd_time = 0;
        wh1_time = 0;
        wh2_time = 0;
        edge_time = 0;
        spmm_time = 0;
        elw_time = 0;
    }
}

template<class SM, class DM>
void gat_forward_layer_wbias(SM *adj,
                       DM *h_in,
                       DM *w, DM *b,
                       DM *a_l,
                       DM *a_r,
                       DM *h_out,
                       DM *wh1,
                       DM *wh2,
                       bool recomp,
                       bool refresh_and_print,
                       int layer,
                       double **&individual_layer_times) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double upd_time = 0.0;
    static double wh1_time = 0.0;
    static double wh2_time = 0.0;
    static double edge_time = 0.0;
    static double spmm_time = 0.0;
    static double elw_time = 0.0;

    DM h_temp;
    DM h_temp2;
#ifdef A_ALLOC
    h_temp.build(h_out->nrows(), h_out->ncols(), h_out->type(), 0);
#else
    h_temp.build(h_out->nrows(), h_out->ncols(), h_out->type());
#endif
    h_temp.set_all(0); // These may be redundant
    h_temp2.set_all(0);
    h_out->set_all(0);

    double t0 = get_time();

    MM_mkl(h_in, w, &h_temp);

    double t1 = get_time();

    /// Calculate WH1
    MM_mkl(&h_temp, a_r, wh1);

    double t2 = get_time();

    /// Calculate WH2
    MM_mkl(&h_temp, a_l, wh2);

    double t3 = get_time();

    /// Edge processing
    auto sum_operator = sum<vT>;
    gSDDVV(adj, wh1, wh2, adj, sum_operator);


    //Apply Edge = Unary ElementWise Operation on Sparse Matrix.
    auto explrelu_operator = explrelu<vT>;
    UEwS(adj, adj, explrelu_operator);


    //SpMV_ones (special case- I don't express it with generalized for extra efficiency).
    DM sums;
    sums.build(h_in->nrows(), 1, h_in->type());
    SpMV_ones(adj, &sums);


    //gSDDVV (maybe not best formulation)
#pragma omp parallel for
    for (iT v = 0; v < adj->nrows(); v++) {
        for (nT e = adj->offset_ptr()[v]; e < adj->offset_ptr()[v + 1]; e++) {
            adj->vals_ptr()[e] = adj->vals_ptr()[e] / sums.vals_ptr()[(nT) v];
        }
    }
    double t4 = get_time();


    /// Node processing
    //Defusing the gSpMM kernel and the Dense element wise Relu yielded acceleration.

    auto wsum_aggr = wsumAgg<dvT, vT, dnT>;

    gSpMM_set<DM, SM>(adj, &h_temp, h_out, wsum_aggr);

    double t5 = get_time();

    DVCBM(b, h_out, sum_operator);

    //Unary element-wise operation on Dense Matrix: ReLU
    auto relu_operator = relu<vT>;
    UEwD<DM>(h_out, h_out, relu_operator);

    double t6 = get_time();
    upd_time += t1 - t0;
    wh1_time += t2 - t1;
    wh2_time += t3 - t2;
    edge_time += t4 - t3;
    spmm_time += t5 - t4;
    elw_time += t6 - t5;

    individual_layer_times[layer][0] += t5 - t4;
    individual_layer_times[layer][1] += t1 - t0;
    individual_layer_times[layer][3] += ((t2 - t1) + (t3 - t2));
    individual_layer_times[layer][4] += t4 - t3;

    h_temp.clear();
    h_temp2.clear();

    if (refresh_and_print) {
        std::cout << "Dense Update took: " << upd_time << std::endl;
        std::cout << "Dense WH1 calculation took: " << wh1_time << std::endl;
        std::cout << "Dense WH2 calculation took: " << wh2_time << std::endl;
        std::cout << "Edge processing took: " << edge_time << std::endl;
        std::cout << "SpMM took: " << spmm_time << std::endl;
        std::cout << "Element wise ReLU on Dense took: " << elw_time << std::endl;
        upd_time = 0;
        wh1_time = 0;
        wh2_time = 0;
        edge_time = 0;
        spmm_time = 0;
        elw_time = 0;
    }
}

template<class SM, class DM>
void gat_forward_layer_split_tiled(std::vector<SM *> adj_vec,
                                   DM *h_in,
                                   DM *w,
                                   DM *a_l,
                                   DM *a_r,
                                   DM *h_out,
                                   DM *wh1,
                                   DM *wh2,
                                   bool recomp,
                                   bool refresh_and_print,
                                   int sparse_segments,
#if defined(ST_1) || defined(ST_2)
        int dense_segments,
#endif
#ifdef LO_KK
        int dense_segments_split,
#endif
                                   int layer,
                                   double **&individual_layer_times) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double upd_time = 0.0;
    static double wh1_time = 0.0;
    static double wh2_time = 0.0;
    static double edge_time = 0.0;
    static double spmm_time = 0.0;
    static double elw_time = 0.0;

    iT sparse_tile_rows = sparse_segments;
#if defined(ST_1) || defined(ST_2)
    iT sparse_tile_cols = dense_segments;
#endif

#ifdef LO_KK
    diT dense_tile_rows = dense_segments_split;
#endif


    DM h_temp;
    DM h_temp2;
#ifdef A_ALLOC
    h_temp.build(h_out->nrows(), h_out->ncols(), h_out->type(), 0);
#else
    h_temp.build(h_out->nrows(), h_out->ncols(), h_out->type());
#endif
    h_out->set_all(0);
    h_temp.set_all(0); // These may be redundant

    double t0 = get_time();

    MM_mkl(h_in, w, &h_temp);

    double t1 = get_time();

    MM_mkl(&h_temp, a_r, wh1);

    double t2 = get_time();

    MM_mkl(&h_temp, a_l, wh2);

    double t3 = get_time();

    //gSDDVV
    // TODO
    auto sum_operator = sum<vT>;
    auto explrelu_operator = explrelu<vT>;
    DM sums;
    sums.build(h_in->nrows(), 1, h_in->type());
    sums.set_all(0);

#if defined(ST_1) || defined(ST_2)
    auto adj = adj_vec.at(0);
    gSDDVV(adj, wh1, wh2, adj, sum_operator);
    UEwS(adj, adj, explrelu_operator);
    SpMV_ones(adj, &sums);

    //gSDDVV (maybe not best formulation)
#pragma omp parallel for
    for (iT v = 0; v < adj->nrows(); v++) {
        for (nT e = adj->offset_ptr()[v]; e < adj->offset_ptr()[v + 1]; e++) {
            adj->vals_ptr()[e] = adj->vals_ptr()[e] / sums.vals_ptr()[(nT) v];
        }
    }
#else
    tile_jj_iip_i_j_kv_sddvv(adj_vec, wh1, wh2, adj_vec, sparse_tile_rows);

    SpMV_ones_vec(adj_vec, &sums);
    for (diT j = 0; j < adj_vec.size(); j += 1) {
        auto adj = adj_vec.at(j);
        iT *adj_row_ptr = adj->row_ids_ptr();
#pragma omp parallel for schedule(dynamic, 8)
        for (iT v_i = 0; v_i < adj->nrows(); v_i++) {
            auto v = adj_row_ptr[v_i];
            auto divis = sums.vals_ptr()[(nT) v];
            for (nT e = adj->offset_ptr()[v_i]; e < adj->offset_ptr()[v_i + 1]; e++) {
                adj->vals_ptr()[e] = adj->vals_ptr()[e] / divis;
            }
        }
    }
#endif
    double t4 = get_time();
    /// Node processing
    //Defusing the gSpMM kernel and the Dense element wise Relu yielded acceleration.

    auto wsum_aggr = wsumAgg<dvT, vT, dnT>;

    // TODO need to make this a parameter(iszed)
    if ((h_in->ncols() < h_out->ncols()) && recomp) {
#ifdef A_ALLOC
        h_temp2.build(h_in->nrows(), h_in->ncols(), h_in->type(), 0);
#else
        h_temp2.build(h_in->nrows(), h_in->ncols(), h_in->type());
#endif
        h_temp2.set_all(0);
        tile_jj_iip_i_j_kv(adj_vec, h_in, &h_temp2, sparse_tile_rows
#if defined(ST_1) || defined(ST_2)
                , dense_segments
#endif
#ifdef LO_KK
                , dense_tile_rows
#endif
        );
        MM_mkl(&h_temp2, w, h_out);
    } else {
        tile_jj_iip_i_j_kv(adj_vec, &h_temp, h_out, sparse_tile_rows
#if defined(ST_1) || defined(ST_2)
                , dense_segments
#endif
#ifdef LO_KK
                , dense_tile_rows
#endif
        );
    }

    double t5 = get_time();

    //Unary element-wise operation on Dense Matrix: ReLU
    auto relu_operator = relu<vT>;
    UEwD<DM>(h_out, h_out, relu_operator);

    double t6 = get_time();
    upd_time += t1 - t0;
    wh1_time += t2 - t1;
    wh2_time += t3 - t2;
    edge_time += t4 - t3;
    spmm_time += t5 - t4;
    elw_time += t6 - t5;

    individual_layer_times[layer][0] += t5 - t4;
    individual_layer_times[layer][1] += t1 - t0;
    individual_layer_times[layer][2] += ((t2 - t1) + (t3 - t2));
    individual_layer_times[layer][3] += t4 - t3;

    h_temp.clear();
    h_temp2.clear();

    if (refresh_and_print) {
        std::cout << "Dense Update took: " << upd_time << std::endl;
        std::cout << "Dense WH1 calculation took: " << wh1_time << std::endl;
        std::cout << "Dense WH2 calculation took: " << wh2_time << std::endl;
        std::cout << "Edge processing took: " << edge_time << std::endl;
        std::cout << "SpMM took: " << spmm_time << std::endl;
        std::cout << "Element wise ReLU on Dense took: " << elw_time << std::endl;
        upd_time = 0;
        wh1_time = 0;
        wh2_time = 0;
        edge_time = 0;
        spmm_time = 0;
        elw_time = 0;
    }
}