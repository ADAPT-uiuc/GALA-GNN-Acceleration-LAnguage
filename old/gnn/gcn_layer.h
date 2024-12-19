
#include "../matrix/csrc_matrix.h"
#include "../matrix/dense_matrix.h"
#include "../utils/threading_utils.h"
#include "../ops/sparse_matrix_ops.h"
#include "../ops/dense_matrix_ops.h"
#include "../ops/aggregators.h"
#include "../ops/uelw_ops.h"
#include "../ops/belw_ops.h"
#include "../ops/enums.h"


template<class SM, class DM, typename itype>
void gcn_forward_layer(SM *adj,
#ifdef GN_2
        DM *ndeg,
#endif
                       DM *h_in,
                       DM *w, DM *b,
                       DM *h_out,
                       GcnOpsOrder order,
                       SpmmVariation spmm_version,
                       itype tile_size,
                       bool refresh_and_print,
                       int layer,
                       double **&individual_layer_times) {

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double matalloc_time = 0.0;
    static double spmm_time = 0.0;
    static double mm_time = 0.0;
    static double elw_time = 0.0;
    auto sum_operator = sum<dvT>;
    static int count_iter = 0;

    switch (order) {
        case gemm_first: {
            // TODO check which time is better - Ask Stefanos
            double start0 = get_time();

            h_out->set_all(0);
            DM HW;
#ifdef A_ALLOC
            HW.build(h_in->nrows(), w->ncols(), h_in->type(), 0);
#else
            HW.build(h_in->nrows(), w->ncols(), h_in->type());
#endif
            HW.set_all(0);

            double start1 = get_time();

            MM_mkl(h_in, w, &HW);

            double start2 = get_time();

#ifdef GN_1
            auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
            gSpMM(adj, &HW, h_out, wsum_aggr);
#else
            auto sum_aggr = sumAgg<dvT, dnT>;
            sumSpMM_scaleout(adj, &HW, h_out, ndeg, sum_aggr);
//
//            sumSpMM(adj, &HW, h_out, sum_aggr);
//            MMbroacast_row(h_out, ndeg, h_out);
#endif

            double start3 = get_time();
            DVCBM(b, h_out, sum_operator);

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start3 - start2;
            individual_layer_times[layer][0] += start3 - start2;

            mm_time += start2 - start1;
            individual_layer_times[layer][1] += start2 - start1;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            HW.clear();
            break;
        }
        case spmm_first: {
            double start0 = get_time();

            DM AH;
#ifdef A_ALLOC
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type(), 0);
#else
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type());
#endif

            AH.set_all(0);

            double start1 = get_time();

#ifdef GN_1
            auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
            gSpMM(adj, h_in, &AH, wsum_aggr);
#else
            auto sum_aggr = sumAgg<dvT, dnT>;
            sumSpMM_scaleout(adj, h_in, &AH, ndeg, sum_aggr);

//            sumSpMM(adj, h_in, &AH, sum_aggr);
//            MMbroacast_row(&AH, ndeg, &AH);
#endif
            double start2 = get_time();

            MMb_mkl(&AH, w, h_out, b);

            double start3 = get_time();

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            // if (count_iter>5){
            //     spmm_time += start2 - start1;
            //     mm_time += start3 - start2;
            //     elw_time += end - start3;
            //     matalloc_time += start1 - start0;
            // }
            // break;
            spmm_time += start2 - start1;
            individual_layer_times[layer][0] += start2 - start1;

            mm_time += start3 - start2;
            individual_layer_times[layer][1] += start3 - start2;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            AH.clear();
            break;
        }
            // count_iter+=1;
    }

    if (refresh_and_print) {

        std::cout << "SpMM took : " << spmm_time << std::endl;
        std::cout << "GeMM took : " << mm_time << std::endl;
        std::cout << "Matrix Allocation took : " << matalloc_time << std::endl;
        std::cout << "Element Wise Unary Dense took : " << elw_time << std::endl;
        spmm_time = 0;
        mm_time = 0;
        matalloc_time = 0;
        elw_time = 0;
        count_iter = 0;
    }
}

template<class SM, class DM, typename itype>
void gcn_forward_layer_gn2(SM *adj,
                           DM *ndeg,
                           DM *h_in,
                           DM *w, DM *b,
                           DM *h_out,
                           GcnOpsOrder order,
                           SpmmVariation spmm_version,
                           itype tile_size,
                           bool refresh_and_print,
                           int layer,
                           double **&individual_layer_times) {

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double matalloc_time = 0.0;
    static double spmm_time = 0.0;
    static double mm_time = 0.0;
    static double elw_time = 0.0;
    auto sum_operator = sum<dvT>;
    static int count_iter = 0;

    switch (order) {
        case gemm_first: {
            // TODO check which time is better - Ask Stefanos
            double start0 = get_time();

            h_out->set_all(0);
            DM HW;

            HW.build(h_in->nrows(), w->ncols(), h_in->type(), 0);
            HW.set_all(0);

            double start1 = get_time();

            MM_mkl(h_in, w, &HW);

            double start2 = get_time();

            auto sum_aggr = sumAgg<dvT, dnT>;
            sumSpMM_scaleout(adj, &HW, h_out, ndeg, sum_aggr);
//
//            sumSpMM(adj, &HW, h_out, sum_aggr);
//            MMbroacast_row(h_out, ndeg, h_out);

            double start3 = get_time();
            DVCBM(b, h_out, sum_operator);

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start3 - start2;
            individual_layer_times[layer][0] += start3 - start2;

            mm_time += start2 - start1;
            individual_layer_times[layer][1] += start2 - start1;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            HW.clear();
            break;
        }
        case spmm_first: {
            double start0 = get_time();

            DM AH;
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type(), 0);

            AH.set_all(0);

            double start1 = get_time();

            auto sum_aggr = sumAgg<dvT, dnT>;
            sumSpMM_scaleout(adj, h_in, &AH, ndeg, sum_aggr);

            double start2 = get_time();

            MMb_mkl(&AH, w, h_out, b);

            double start3 = get_time();

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            // if (count_iter>5){
            //     spmm_time += start2 - start1;
            //     mm_time += start3 - start2;
            //     elw_time += end - start3;
            //     matalloc_time += start1 - start0;
            // }
            // break;
            spmm_time += start2 - start1;
            individual_layer_times[layer][0] += start2 - start1;

            mm_time += start3 - start2;
            individual_layer_times[layer][1] += start3 - start2;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            AH.clear();
            break;
        }
            // count_iter+=1;
    }

    if (refresh_and_print) {

        std::cout << "SpMM took : " << spmm_time << std::endl;
        std::cout << "GeMM took : " << mm_time << std::endl;
        std::cout << "Matrix Allocation took : " << matalloc_time << std::endl;
        std::cout << "Element Wise Unary Dense took : " << elw_time << std::endl;
        spmm_time = 0;
        mm_time = 0;
        matalloc_time = 0;
        elw_time = 0;
        count_iter = 0;
    }
}

template<class SM, class DM, typename itype>
void gcn_forward_layer_eft(SM *adj,
                           DM *h_in,
                           DM *w, DM *b,
                           DM *h_out,
                           GcnOpsOrder order,
                           SpmmVariation spmm_version,
                           itype tile_size,
                           bool refresh_and_print,
                           int layer,
                           double **&individual_layer_times,
                           typename SM::itype skip_v) {

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double matalloc_time = 0.0;
    static double spmm_time = 0.0;
    static double mm_time = 0.0;
    static double elw_time = 0.0;
    auto sum_operator = sum<dvT>;
    static int count_iter = 0;

    switch (order) {
        case gemm_first: {
            double start0 = get_time();

            h_out->set_all(0);
            DM HW;
            HW.build(h_in->nrows(), w->ncols(), h_in->type(), 0);
            HW.set_all(0);

            double start1 = get_time();

            MM_mkl(h_in, w, &HW);

            double start2 = get_time();

            auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
            gSpMM_skip(adj, &HW, h_out, wsum_aggr, skip_v);

            double start3 = get_time();
            DVCBM(b, h_out, sum_operator);

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start3 - start2;
            individual_layer_times[layer][0] += start3 - start2;

            mm_time += start2 - start1;
            individual_layer_times[layer][1] += start2 - start1;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            HW.clear();
            break;
        }
        case spmm_first: {
            double start0 = get_time();

            DM AH;
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type(), 0);

            AH.set_all(0);

            double start1 = get_time();

            auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
            gSpMM_skip(adj, h_in, &AH, wsum_aggr, skip_v);
            double start2 = get_time();

            MMb_mkl(&AH, w, h_out, b);

            double start3 = get_time();

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start2 - start1;
            individual_layer_times[layer][0] += start2 - start1;

            mm_time += start3 - start2;
            individual_layer_times[layer][1] += start3 - start2;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            AH.clear();
            break;
        }
    }

    if (refresh_and_print) {

        std::cout << "SpMM took : " << spmm_time << std::endl;
        std::cout << "GeMM took : " << mm_time << std::endl;
        std::cout << "Matrix Allocation took : " << matalloc_time << std::endl;
        std::cout << "Element Wise Unary Dense took : " << elw_time << std::endl;
        spmm_time = 0;
        mm_time = 0;
        matalloc_time = 0;
        elw_time = 0;
        count_iter = 0;
    }
}

#ifdef MKL

template<class SM, class DM, typename itype>
void gcn_forward_layer_mkl(SM *adj,
#ifdef GN_2
        DM *ndeg,
#endif
                           DM *h_in,
                           DM *w, DM *b,
                           DM *h_out,
                           GcnOpsOrder order,
                           SpmmVariation spmm_version,
                           itype tile_size,
                           bool refresh_and_print,
                           int layer,
                           double **&individual_layer_times,
                           sparse_matrix_t *A,
                           matrix_descr descrA) {

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double matalloc_time = 0.0;
    static double spmm_time = 0.0;
    static double mm_time = 0.0;
    static double elw_time = 0.0;
    auto sum_operator = sum<dvT>;
    static int count_iter = 0;

    switch (order) {
        case gemm_first: {
            // TODO check which time is better - Ask Stefanos
            double start0 = get_time();

            h_out->set_all(0);
            DM HW;
#ifdef A_ALLOC
            HW.build(h_in->nrows(), w->ncols(), h_in->type(), 0);
#else
            HW.build(h_in->nrows(), w->ncols(), h_in->type());
#endif
            HW.set_all(0);

            double start1 = get_time();

            MM_mkl(h_in, w, &HW);

            double start2 = get_time();

            mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                            1,
                            *A,
                            descrA,
                            SPARSE_LAYOUT_ROW_MAJOR,
                            HW.vals_ptr(),
                            h_out->ncols(),
                            HW.ncols(),
                            0,
                            h_out->vals_ptr(),
                            h_out->ncols());

            double start3 = get_time();
            DVCBM(b, h_out, sum_operator);

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start3 - start2;
            individual_layer_times[layer][0] += start3 - start2;

            mm_time += start2 - start1;
            individual_layer_times[layer][1] += start2 - start1;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            HW.clear();
            break;
        }
        case spmm_first: {
            double start0 = get_time();

            DM AH;
#ifdef A_ALLOC
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type(), 0);
#else
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type());
#endif

            AH.set_all(0);

            double start1 = get_time();

            mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                            1,
                            *A,
                            descrA,
                            SPARSE_LAYOUT_ROW_MAJOR,
                            h_in->vals_ptr(),
                            AH.ncols(),
                            h_in->ncols(),
                            0,
                            AH.vals_ptr(),
                            AH.ncols());

            double start2 = get_time();

            MMb_mkl(&AH, w, h_out, b);

            double start3 = get_time();

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            // if (count_iter>5){
            //     spmm_time += start2 - start1;
            //     mm_time += start3 - start2;
            //     elw_time += end - start3;
            //     matalloc_time += start1 - start0;
            // }
            // break;
            spmm_time += start2 - start1;
            individual_layer_times[layer][0] += start2 - start1;

            mm_time += start3 - start2;
            individual_layer_times[layer][1] += start3 - start2;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            AH.clear();
            break;
        }
            // count_iter+=1;
    }

    if (refresh_and_print) {

        std::cout << "SpMM took : " << spmm_time << std::endl;
        std::cout << "GeMM took : " << mm_time << std::endl;
        std::cout << "Matrix Allocation took : " << matalloc_time << std::endl;
        std::cout << "Element Wise Unary Dense took : " << elw_time << std::endl;
        spmm_time = 0;
        mm_time = 0;
        matalloc_time = 0;
        elw_time = 0;
        count_iter = 0;
    }
}

#endif

template<class SM, class DM, typename itype>
void gcn_forward_layer_tiled(SM *adj,
                             DM *h_in,
                             DM *w, DM *b,
                             DM *h_out,
                             GcnOpsOrder order,
                             SpmmVariation spmm_version,
                             itype tile_size,
                             bool refresh_and_print,
                             int sparse_segments, int dense_segments,
                             int layer,
                             double **&individual_layer_times) {

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double matalloc_time = 0.0;
    static double spmm_time = 0.0;
    static double mm_time = 0.0;
    static double elw_time = 0.0;
    auto sum_operator = sum<dvT>;
    static int count_iter = 0;

    iT n = adj->nrows();
    iT sparse_tile_rows = sparse_segments;
    diT dense_tile_rows = dense_segments;

    // TODO would this cause a OOM on stack?
//     nT* copy_offsets = (nT*)alloca((adj->nrows() + 1) * sizeof(nT));
// //    std::cout << "Comes here" << std::endl;
//     memcpy(copy_offsets, adj->offset_ptr(), (adj->nrows() + 1) * sizeof(nT));
//    std::cout << "Comes here" << std::endl;

    switch (order) {
        case gemm_first: {
            double start0 = get_time();

            h_out->set_all(0);
            DM HW;
#ifdef A_ALLOC
            HW.build(h_in->nrows(), w->ncols(), h_in->type(), 0);
#else
            HW.build(h_in->nrows(), w->ncols(), h_in->type());
#endif
            HW.set_all(0);

            double start1 = get_time();
            MM_mkl(h_in, w, &HW);

            double start2 = get_time();

            auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
            for (diT j = 0; j < HW.nrows(); j += dense_tile_rows) {
                diT drows_end = std::min(j + dense_tile_rows, HW.nrows());
#pragma omp parallel for schedule(dynamic, 1)
                for (iT i = 0; i < adj->nrows(); i += sparse_tile_rows) {
                    GNOpTile<SM, DM> tile_info;

                    tile_info.srows_start = i;
                    tile_info.srows_end = std::min(i + sparse_tile_rows, adj->nrows());
                    tile_info.drows_start = j;
                    tile_info.drows_end = drows_end;

                    gSpMM_tiled(adj, &HW, h_out, wsum_aggr, &tile_info, adj->offset_ptr());
                }
            }

            double start3 = get_time();

            DVCBM(b, h_out, sum_operator);

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start3 - start2;
            mm_time += start2 - start1;
            elw_time += end - start3;
            matalloc_time += start1 - start0;

            break;
        }
        case spmm_first: {
            double start0 = get_time();

            DM AH;
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type());
            AH.set_all(0);

            double start1 = get_time();

            auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
            for (diT j = 0; j < h_in->nrows(); j += dense_tile_rows) {
                diT drows_end = std::min(j + dense_tile_rows, h_in->nrows());
//#pragma omp parallel for schedule(guided)
#pragma omp parallel for schedule(dynamic, 1)
                for (iT i = 0; i < adj->nrows(); i += sparse_tile_rows) {
                    GNOpTile<SM, DM> tile_info;

                    tile_info.srows_start = i;
                    tile_info.srows_end = std::min(i + sparse_tile_rows, adj->nrows());
                    tile_info.drows_start = j;
                    tile_info.drows_end = drows_end;

                    gSpMM_tiled(adj, h_in, &AH, wsum_aggr, &tile_info, adj->offset_ptr());
                }
            }


            double start2 = get_time();


            MMb_mkl(&AH, w, h_out, b);

            double start3 = get_time();

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start2 - start1;
            mm_time += start3 - start2;
            elw_time += end - start3;
            matalloc_time += start1 - start0;

            break;
        }
    }

    if (refresh_and_print) {

        std::cout << "SpMM took1 : " << spmm_time << std::endl;
        std::cout << "GeMM took1 : " << mm_time << std::endl;
        std::cout << "Matrix Allocation took1 : " << matalloc_time << std::endl;
        std::cout << "Element Wise Unary Dense took1 : " << elw_time << std::endl;
        // print_count();
        spmm_time = 0;
        mm_time = 0;
        matalloc_time = 0;
        elw_time = 0;
    }
}

template<class SM, class DM, typename itype>
void gcn_forward_layer_dgl_tiled(SM *adj,
                                 DM *h_in,
                                 DM *w, DM *b,
                                 DM *h_out,
                                 GcnOpsOrder order, SpmmVariation spmm_version, itype tile_size, bool refresh_and_print,
                                 int sparse_segments, int dense_segments) {

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double matalloc_time = 0.0;
    static double spmm_time = 0.0;
    static double mm_time = 0.0;
    static double elw_time = 0.0;
    auto sum_operator = sum<dvT>;

    iT n = adj->nrows();
    iT sparse_tile_rows = sparse_segments;
    diT dense_tile_rows = dense_segments;

    // TODO would this cause a OOM on stack?
    nT *copy_offsets = (nT *) alloca((adj->nrows() + 1) * sizeof(nT));
    memcpy(copy_offsets, adj->offset_ptr(), (adj->nrows() + 1) * sizeof(nT));
//    std::cout << "------------" << std::endl;
//    std::cout << adj->offset_ptr()[adj->nrows() - 1] << " " << copy_offsets[adj->nrows() - 1] << std::endl;
//    std::cout << adj->offset_ptr()[adj->nrows()] << " " << copy_offsets[adj->nrows()] << std::endl;

    switch (order) {
        case gemm_first: {
            double start0 = get_time();

            h_out->set_all(0);
            DM HW;
            HW.build(h_in->nrows(), w->ncols(), h_in->type());
            HW.set_all(0);

            double start1 = get_time();
            MM_mkl(h_in, w, &HW);

            double start2 = get_time();

            auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
            for (diT j = 0; j < HW.nrows(); j += dense_tile_rows) {
                diT drows_end = std::min(j + dense_tile_rows, HW.nrows());
//#pragma omp parallel for schedule(guided)
#pragma omp parallel for schedule(dynamic, 1)
                for (iT i = 0; i < adj->nrows(); i += sparse_tile_rows) {
                    GNOpTile<SM, DM> tile_info;

                    tile_info.srows_start = i;
                    tile_info.srows_end = std::min(i + sparse_tile_rows, adj->nrows());
                    tile_info.drows_start = j;
                    tile_info.drows_end = drows_end;

                    gSpMM_dgl_tiled(adj,
                                    &HW,
                                    h_out,
                                    wsum_aggr,
                                    &tile_info,
                                    copy_offsets);
                }
            }

            double start3 = get_time();

            DVCBM(b, h_out, sum_operator);

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start3 - start2;
            mm_time += start2 - start1;
            elw_time += end - start3;
            matalloc_time += start1 - start0;

            break;
        }
        case spmm_first: {
            double start0 = get_time();

            DM AH;
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type());
            AH.set_all(0);

            double start1 = get_time();

            auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
            for (diT j = 0; j < h_in->nrows(); j += dense_tile_rows) {
                diT drows_end = std::min(j + dense_tile_rows, h_in->nrows());
//#pragma omp parallel for schedule(guided)
#pragma omp parallel for schedule(dynamic, 1)
                for (iT i = 0; i < adj->nrows(); i += sparse_tile_rows) {
                    GNOpTile<SM, DM> tile_info;

                    tile_info.srows_start = i;
                    tile_info.srows_end = std::min(i + sparse_tile_rows, adj->nrows());
                    tile_info.drows_start = j;
                    tile_info.drows_end = drows_end;

                    gSpMM_dgl_tiled(adj, h_in, &AH, wsum_aggr, &tile_info, copy_offsets);
                }
            }


            double start2 = get_time();


            MMb_mkl(&AH, w, h_out, b);

            double start3 = get_time();

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start2 - start1;
            mm_time += start3 - start2;
            elw_time += end - start3;
            matalloc_time += start1 - start0;

            break;
        }
    }

    if (refresh_and_print) {
        std::cout << "SpMM took1 : " << spmm_time << std::endl;
        std::cout << "GeMM took1 : " << mm_time << std::endl;
        std::cout << "Matrix Allocation took1 : " << matalloc_time << std::endl;
        std::cout << "Element Wise Unary Dense took1 : " << elw_time << std::endl;
        // print_count();
        spmm_time = 0;
        mm_time = 0;
        matalloc_time = 0;
        elw_time = 0;
    }
}

template<class SM, class DM, typename itype>
void gcn_forward_layer_slice_dgl_tiled(SM *adj,
                                       DM *h_in,
                                       DM *w, DM *b,
                                       DM *h_out,
                                       GcnOpsOrder order, SpmmVariation spmm_version, itype tile_size,
                                       bool refresh_and_print,
                                       int sparse_segments, int dense_segments,
                                       int dense_feats_per_tile) {

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double matalloc_time = 0.0;
    static double spmm_time = 0.0;
    static double mm_time = 0.0;
    static double elw_time = 0.0;
    auto sum_operator = sum<dvT>;

    iT n = adj->nrows();
    iT sparse_tile_rows = sparse_segments;
    diT dense_tile_rows = dense_segments;

    diT dense_tile_cols = dense_feats_per_tile;

    // TODO would this cause a OOM on stack?
    nT *copy_offsets = (nT *) alloca((adj->nrows() + 1) * sizeof(nT));
    memcpy(copy_offsets, adj->offset_ptr(), (adj->nrows() + 1) * sizeof(nT));

    nT *copy_offsets_k = (nT *) alloca((adj->nrows() + 1) * sizeof(nT));
//    std::cout << "------------" << std::endl;
//    std::cout << adj->offset_ptr()[adj->nrows() - 1] << " " << copy_offsets[adj->nrows() - 1] << std::endl;
//    std::cout << adj->offset_ptr()[adj->nrows()] << " " << copy_offsets[adj->nrows()] << std::endl;

    switch (order) {
        case gemm_first: {
            double start0 = get_time();

            h_out->set_all(0);
            DM HW;
            HW.build(h_in->nrows(), w->ncols(), h_in->type());
            HW.set_all(0);

            double start1 = get_time();
            MM_mkl(h_in, w, &HW);

            double start2 = get_time();

            auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
            for (diT k = 0; k < HW.ncols(); k += dense_tile_cols) {
                diT dcols_len = std::min(dense_tile_cols, HW.ncols() - k);
                memcpy(copy_offsets_k, copy_offsets, (adj->nrows() + 1) * sizeof(nT));
                for (diT j = 0; j < HW.nrows(); j += dense_tile_rows) {
                    diT drows_end = std::min(j + dense_tile_rows, HW.nrows());
//#pragma omp parallel for schedule(guided)
#pragma omp parallel for schedule(dynamic, 1)
                    for (iT i = 0; i < adj->nrows(); i += sparse_tile_rows) {
                        GNOpTile<SM, DM> tile_info;

                        tile_info.srows_start = i;
                        tile_info.srows_end = std::min(i + sparse_tile_rows, adj->nrows());
                        tile_info.drows_start = j;
                        tile_info.drows_end = drows_end;

                        gSpMM_slice_dgl_tiled(adj,
                                              &HW,
                                              h_out,
                                              wsum_aggr,
                                              &tile_info,
                                              copy_offsets_k,
                                              k,
                                              dcols_len);
                    }
                }
            }
            memcpy(copy_offsets, copy_offsets_k, (adj->nrows() + 1) * sizeof(nT));

            double start3 = get_time();

            DVCBM(b, h_out, sum_operator);

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start3 - start2;
            mm_time += start2 - start1;
            elw_time += end - start3;
            matalloc_time += start1 - start0;

            break;
        }
        case spmm_first: {
            double start0 = get_time();

            DM AH;
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type());
            AH.set_all(0);

            double start1 = get_time();

            auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
            for (diT k = 0; k < h_in->ncols(); k += dense_tile_cols) {
                diT dcols_len = std::min(dense_tile_cols, h_in->ncols() - k);
                memcpy(copy_offsets_k, copy_offsets, (adj->nrows() + 1) * sizeof(nT));
                for (diT j = 0; j < h_in->nrows(); j += dense_tile_rows) {
                    diT drows_end = std::min(j + dense_tile_rows, h_in->nrows());
//#pragma omp parallel for schedule(guided)
#pragma omp parallel for schedule(dynamic, 1)
                    for (iT i = 0; i < adj->nrows(); i += sparse_tile_rows) {
                        GNOpTile<SM, DM> tile_info;

                        tile_info.srows_start = i;
                        tile_info.srows_end = std::min(i + sparse_tile_rows, adj->nrows());
                        tile_info.drows_start = j;
                        tile_info.drows_end = drows_end;

                        gSpMM_slice_dgl_tiled(adj,
                                              h_in,
                                              &AH,
                                              wsum_aggr,
                                              &tile_info,
                                              copy_offsets_k,
                                              k,
                                              dcols_len);
                    }
                }
            }
            memcpy(copy_offsets, copy_offsets_k, (adj->nrows() + 1) * sizeof(nT));

            double start2 = get_time();


            MMb_mkl(&AH, w, h_out, b);

            double start3 = get_time();

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start2 - start1;
            mm_time += start3 - start2;
            elw_time += end - start3;
            matalloc_time += start1 - start0;

            break;
        }
    }

    if (refresh_and_print) {

        std::cout << "SpMM took1 : " << spmm_time << std::endl;
        std::cout << "GeMM took1 : " << mm_time << std::endl;
        std::cout << "Matrix Allocation took1 : " << matalloc_time << std::endl;
        std::cout << "Element Wise Unary Dense took1 : " << elw_time << std::endl;
        // print_count();
        spmm_time = 0;
        mm_time = 0;
        matalloc_time = 0;
        elw_time = 0;
    }
}

template<class SM, class DM>
void gcn_forward_layer_split_tiled(std::vector<SM *> adj_vec,
#ifdef GN_2
        DM *ndeg,
#endif
                                   DM *h_in,
                                   DM *w, DM *b,
                                   DM *h_out,
                                   GcnOpsOrder order,
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

    static double matalloc_time = 0.0;
    static double spmm_time = 0.0;
    static double mm_time = 0.0;
    static double elw_time = 0.0;
    auto sum_operator = sum<dvT>;

    iT sparse_tile_rows = sparse_segments;
#if defined(ST_1) || defined(ST_2)
    iT sparse_tile_cols = dense_segments;
#endif

#ifdef LO_KK
    //    std::cout << "Slice is: " << dense_segments << std::endl;
        diT dense_tile_rows = dense_segments_split;
#endif

    switch (order) {
        case gemm_first: {
            double start0 = get_time();

            h_out->set_all(0);
            DM HW;
#ifdef A_ALLOC
            HW.build(h_in->nrows(), w->ncols(), h_in->type(), 0);
#else
            HW.build(h_in->nrows(), w->ncols(), h_in->type());
#endif
            HW.set_all(0);

            double start1 = get_time();
#ifdef MM_IMP
            MM(h_in, w, &HW);
#else
            MM_mkl(h_in, w, &HW);
#endif
            double start2 = get_time();

#ifdef LO_1
            tile_ii_jj_i_j_kv(adj_vec, &HW, h_out, sparse_tile_rows);
#elif defined(LO_2) || defined(LO_3) || defined(LO_5)
            tile_jj_iip_i_j_kv(adj_vec, &HW, h_out, sparse_tile_rows
#if defined(ST_1) || defined(ST_2)
                    , dense_segments
#endif
#ifdef LO_KK
                    , dense_tile_rows
#endif
            );
#elif LO_10
            tile_jj_iip_i_j_kv(adj_vec, &HW, h_out, sparse_tile_rows);
#elif LO_11
            tile_jj_iip_i_j_kv(adj_vec, &HW, h_out, sparse_tile_rows);
#elif LO_12
            tile_jj_kk_ii_i_j_k(adj_vec, &HW, h_out, sparse_tile_rows, dense_segments);
#endif

#ifdef GN_2
            MMbroacast_row(h_out, ndeg, h_out);
#endif
            double start3 = get_time();

            DVCBM(b, h_out, sum_operator);

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start3 - start2;
            individual_layer_times[layer][0] += start3 - start2;

            mm_time += start2 - start1;
            individual_layer_times[layer][1] += start2 - start1;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            HW.clear();
            break;
        }
        case spmm_first: {
            double start0 = get_time();

            DM AH;
#ifdef A_ALLOC
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type(), 0);
#else
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type());
#endif
            AH.set_all(0);

            double start1 = get_time();

#ifdef AU_C
#ifdef AUT_4
            tile_jj_iip_i_j_kv_p_weight(adj_vec, h_in, w, b, &AH, h_out, sparse_tile_rows);
            double start2 = get_time();
            double start3 = get_time();
#else
            tile_jj_iip_i_j_kv_x_weight(adj_vec, h_in, w, b, &AH, h_out, sparse_tile_rows);
            double start2 = get_time();
//            MM_mkl_additive(&AH, w, h_out);
//            MMb_mkl(&AH, w, h_out, b);
//            MM_additive(&AH, w, h_out);
            double start3 = get_time();
#endif
#else

#ifdef LO_1
            tile_ii_jj_i_j_kv(adj_vec, h_in, &AH, sparse_tile_rows);
#elif defined(LO_2) || defined(LO_3) || defined(LO_5)
            tile_jj_iip_i_j_kv(adj_vec, h_in, &AH, sparse_tile_rows
#if defined(ST_1) || defined(ST_2)
                    , dense_segments
#endif
#ifdef LO_KK
                    , dense_tile_rows
#endif
            );
#elif LO_10
            tile_jj_iip_i_j_kv(adj_vec, h_in, &AH, sparse_tile_rows);
#elif LO_11
            tile_jj_iip_i_j_kv(adj_vec, h_in, &AH, sparse_tile_rows);
#elif LO_12
            tile_jj_kk_ii_i_j_k(adj_vec, h_in, &AH, sparse_tile_rows, dense_segments);
#endif

#ifdef GN_2
            MMbroacast_row(&AH, ndeg, &AH);
#endif

            double start2 = get_time();

#ifdef MM_IMP
            MMb(&AH, w, h_out, b);
#else
            MMb_mkl(&AH, w, h_out, b);
#endif
            double start3 = get_time();
#endif

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start2 - start1;
            individual_layer_times[layer][0] += start2 - start1;

            mm_time += start3 - start2;
            individual_layer_times[layer][1] += start3 - start2;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            AH.clear();
            break;
        }
    }

    if (refresh_and_print) {

        std::cout << "SpMM took1 : " << spmm_time << std::endl;
        std::cout << "GeMM took1 : " << mm_time << std::endl;
        std::cout << "Matrix Allocation took1 : " << matalloc_time << std::endl;
        std::cout << "Element Wise Unary Dense took1 : " << elw_time << std::endl;
        // print_count();
        spmm_time = 0;
        mm_time = 0;
        matalloc_time = 0;
        elw_time = 0;
    }
}

template<class SM, class DM>
void gcn_forward_layer_split_tiled_gn2(std::vector<SM *> adj_vec,
                                       DM *ndeg,
                                       DM *h_in,
                                       DM *w, DM *b,
                                       DM *h_out,
                                       GcnOpsOrder order,
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

    static double matalloc_time = 0.0;
    static double spmm_time = 0.0;
    static double mm_time = 0.0;
    static double elw_time = 0.0;
    auto sum_operator = sum<dvT>;

    iT sparse_tile_rows = sparse_segments;
#if defined(ST_1) || defined(ST_2)
    iT sparse_tile_cols = dense_segments;
#endif

#ifdef LO_KK
    //    std::cout << "Slice is: " << dense_segments << std::endl;
        diT dense_tile_rows = dense_segments_split;
#endif

    switch (order) {
        case gemm_first: {
            double start0 = get_time();

            h_out->set_all(0);
            DM HW;
#ifdef A_ALLOC
            HW.build(h_in->nrows(), w->ncols(), h_in->type(), 0);
#else
            HW.build(h_in->nrows(), w->ncols(), h_in->type());
#endif
            HW.set_all(0);

            double start1 = get_time();
#ifdef MM_IMP
            MM(h_in, w, &HW);
#else
            MM_mkl(h_in, w, &HW);
#endif
            double start2 = get_time();

#ifdef LO_1
            tile_ii_jj_i_j_kv(adj_vec, &HW, h_out, sparse_tile_rows);
#elif defined(LO_2) || defined(LO_3) || defined(LO_5)
            tile_jj_iip_i_j_kv_gn2(adj_vec, &HW, h_out, sparse_tile_rows
#if defined(ST_1) || defined(ST_2)
                    , dense_segments
#endif
#ifdef LO_KK
                    , dense_tile_rows
#endif
            );
#elif LO_10
            tile_jj_iip_i_j_kv(adj_vec, &HW, h_out, sparse_tile_rows);
#elif LO_11
            tile_jj_iip_i_j_kv(adj_vec, &HW, h_out, sparse_tile_rows);
#elif LO_12
            tile_jj_kk_ii_i_j_k(adj_vec, &HW, h_out, sparse_tile_rows, dense_segments);
#endif

            MMbroacast_row(h_out, ndeg, h_out);
            double start3 = get_time();

            DVCBM(b, h_out, sum_operator);

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start3 - start2;
            individual_layer_times[layer][0] += start3 - start2;

            mm_time += start2 - start1;
            individual_layer_times[layer][1] += start2 - start1;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            HW.clear();
            break;
        }
        case spmm_first: {
            double start0 = get_time();

            DM AH;
#ifdef A_ALLOC
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type(), 0);
#else
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type());
#endif
            AH.set_all(0);

            double start1 = get_time();

#ifdef AU_C
#ifdef AUT_4
            tile_jj_iip_i_j_kv_p_weight(adj_vec, h_in, w, b, &AH, h_out, sparse_tile_rows);
            double start2 = get_time();
            double start3 = get_time();
#else
            tile_jj_iip_i_j_kv_x_weight(adj_vec, h_in, w, b, &AH, h_out, sparse_tile_rows);
            double start2 = get_time();
//            MM_mkl_additive(&AH, w, h_out);
//            MMb_mkl(&AH, w, h_out, b);
//            MM_additive(&AH, w, h_out);
            double start3 = get_time();
#endif
#else

#ifdef LO_1
            tile_ii_jj_i_j_kv(adj_vec, h_in, &AH, sparse_tile_rows);
#elif defined(LO_2) || defined(LO_3) || defined(LO_5)
            tile_jj_iip_i_j_kv_gn2(adj_vec, h_in, &AH, sparse_tile_rows
#if defined(ST_1) || defined(ST_2)
                    , dense_segments
#endif
#ifdef LO_KK
                    , dense_tile_rows
#endif
            );
#elif LO_10
            tile_jj_iip_i_j_kv(adj_vec, h_in, &AH, sparse_tile_rows);
#elif LO_11
            tile_jj_iip_i_j_kv(adj_vec, h_in, &AH, sparse_tile_rows);
#elif LO_12
            tile_jj_kk_ii_i_j_k(adj_vec, h_in, &AH, sparse_tile_rows, dense_segments);
#endif

            MMbroacast_row(&AH, ndeg, &AH);
            double start2 = get_time();

#ifdef MM_IMP
            MMb(&AH, w, h_out, b);
#else
            MMb_mkl(&AH, w, h_out, b);
#endif
            double start3 = get_time();
#endif

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start2 - start1;
            individual_layer_times[layer][0] += start2 - start1;

            mm_time += start3 - start2;
            individual_layer_times[layer][1] += start3 - start2;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            AH.clear();
            break;
        }
    }

    if (refresh_and_print) {

        std::cout << "SpMM took1 : " << spmm_time << std::endl;
        std::cout << "GeMM took1 : " << mm_time << std::endl;
        std::cout << "Matrix Allocation took1 : " << matalloc_time << std::endl;
        std::cout << "Element Wise Unary Dense took1 : " << elw_time << std::endl;
        // print_count();
        spmm_time = 0;
        mm_time = 0;
        matalloc_time = 0;
        elw_time = 0;
    }
}

template<class SM, class DM>
void gcn_forward_layer_split_tiled(std::vector<SM *> adj_vec,
                                   DM *h_in,
                                   DM *w, DM *b,
                                   DM *h_out,
                                   GcnOpsOrder order,
                                   bool refresh_and_print,
                                   std::vector<std::vector<GNOpTile<SM, DM>>> &tile_infos,
                                   int layer,
                                   double **&individual_layer_times) {

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double matalloc_time = 0.0;
    static double spmm_time = 0.0;
    static double mm_time = 0.0;
    static double elw_time = 0.0;
    auto sum_operator = sum<dvT>;

    switch (order) {
        case gemm_first: {
            double start0 = get_time();

            h_out->set_all(0);
            DM HW;
#ifdef A_ALLOC
            HW.build(h_in->nrows(), w->ncols(), h_in->type(), 0);
#else
            HW.build(h_in->nrows(), w->ncols(), h_in->type());
#endif
            HW.set_all(0);

            double start1 = get_time();
#ifdef MM_IMP
            MM(h_in, w, &HW);
#else
            MM_mkl(h_in, w, &HW);
#endif
            double start2 = get_time();

            variable_tile_jj_iip_i_j_kv(adj_vec, &HW, h_out, tile_infos);

            double start3 = get_time();

            DVCBM(b, h_out, sum_operator);

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start3 - start2;
            individual_layer_times[layer][0] += start3 - start2;

            mm_time += start2 - start1;
            individual_layer_times[layer][1] += start2 - start1;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            HW.clear();
            break;
        }
        case spmm_first: {
            double start0 = get_time();

            DM AH;
#ifdef A_ALLOC
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type(), 0);
#else
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type());
#endif
            AH.set_all(0);

            double start1 = get_time();

            variable_tile_jj_iip_i_j_kv(adj_vec, h_in, &AH, tile_infos);

            double start2 = get_time();

#ifdef MM_IMP
            MMb(&AH, w, h_out, b);
#else
            MMb_mkl(&AH, w, h_out, b);
#endif
            double start3 = get_time();

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start2 - start1;
            individual_layer_times[layer][0] += start2 - start1;

            mm_time += start3 - start2;
            individual_layer_times[layer][1] += start3 - start2;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            AH.clear();
            break;
        }
    }

    if (refresh_and_print) {

        std::cout << "SpMM took1 : " << spmm_time << std::endl;
        std::cout << "GeMM took1 : " << mm_time << std::endl;
        std::cout << "Matrix Allocation took1 : " << matalloc_time << std::endl;
        std::cout << "Element Wise Unary Dense took1 : " << elw_time << std::endl;
        // print_count();
        spmm_time = 0;
        mm_time = 0;
        matalloc_time = 0;
        elw_time = 0;
    }
}

template<class SM, class DM>
void gcn_forward_layer_split_segmented_tiled(std::vector<SM *> adj_vec,
                                             DM *h_in,
                                             DM *w, DM *b,
                                             DM *h_out,
                                             GcnOpsOrder order,
                                             bool refresh_and_print,
                                             int sparse_row_size,
                                             int sparse_segment_size,
                                             int layer,
                                             double **&individual_layer_times) {

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double matalloc_time = 0.0;
    static double spmm_time = 0.0;
    static double mm_time = 0.0;
    static double elw_time = 0.0;
    auto sum_operator = sum<dvT>;

    iT sparse_tile_rows = sparse_row_size;
    iT sparse_segment_tiles = sparse_segment_size;

    switch (order) {
        case gemm_first: {
            double start0 = get_time();

            h_out->set_all(0);
            DM HW;
            HW.build(h_in->nrows(), w->ncols(), h_in->type());
            HW.set_all(0);

            double start1 = get_time();
            MM_mkl(h_in, w, &HW);

            double start2 = get_time();

            tile_seg_iip_jj_i_j_kv(adj_vec, &HW, h_out,
                                   sparse_tile_rows, sparse_segment_tiles);

            double start3 = get_time();

            DVCBM(b, h_out, sum_operator);

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start3 - start2;
            individual_layer_times[layer][0] += start3 - start2;

            mm_time += start2 - start1;
            individual_layer_times[layer][1] += start2 - start1;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            HW.clear();
            break;
        }
        case spmm_first: {
            double start0 = get_time();

            DM AH;
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type());
            AH.set_all(0);

            double start1 = get_time();

            tile_seg_iip_jj_i_j_kv(adj_vec, h_in, &AH,
                                   sparse_tile_rows, sparse_segment_tiles);

            double start2 = get_time();


            MMb_mkl(&AH, w, h_out, b);

            double start3 = get_time();

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start2 - start1;
            individual_layer_times[layer][0] += start2 - start1;

            mm_time += start3 - start2;
            individual_layer_times[layer][1] += start3 - start2;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            AH.clear();
            break;
        }
    }

    if (refresh_and_print) {

        std::cout << "SpMM took1 : " << spmm_time << std::endl;
        std::cout << "GeMM took1 : " << mm_time << std::endl;
        std::cout << "Matrix Allocation took1 : " << matalloc_time << std::endl;
        std::cout << "Element Wise Unary Dense took1 : " << elw_time << std::endl;
        // print_count();
        spmm_time = 0;
        mm_time = 0;
        matalloc_time = 0;
        elw_time = 0;
    }
}

template<class SM, class DM>
void gcn_forward_layer_slice_split_tiled(std::vector<SM *> adj_vec,
                                         DM *h_in,
                                         DM *w, DM *b,
                                         DM *h_out,
                                         GcnOpsOrder order,
                                         bool refresh_and_print,
                                         int sparse_segments,
                                         int dense_feats_per_tile,
                                         int layer,
                                         double **&individual_layer_times) {

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    static double matalloc_time = 0.0;
    static double spmm_time = 0.0;
    static double mm_time = 0.0;
    static double elw_time = 0.0;
    auto sum_operator = sum<dvT>;

    iT sparse_tile_rows = sparse_segments;
    diT dense_tile_cols = dense_feats_per_tile;

    switch (order) {
        case gemm_first: {
            double start0 = get_time();

            h_out->set_all(0);
            DM HW;
            HW.build(h_in->nrows(), w->ncols(), h_in->type());
            HW.set_all(0);

            double start1 = get_time();
            MM_mkl(h_in, w, &HW);

            double start2 = get_time();

            tile_kk_jj_iip_i_j_kv(adj_vec, &HW, h_out, sparse_tile_rows, dense_tile_cols);

            double start3 = get_time();

            DVCBM(b, h_out, sum_operator);

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start3 - start2;
            individual_layer_times[layer][0] += start3 - start2;

            mm_time += start2 - start1;
            individual_layer_times[layer][1] += start2 - start1;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            HW.clear();
            break;
        }
        case spmm_first: {
            double start0 = get_time();

            DM AH;
            AH.build(h_in->nrows(), h_in->ncols(), h_in->type());
            AH.set_all(0);

            double start1 = get_time();

            tile_kk_jj_iip_i_j_kv(adj_vec, h_in, &AH, sparse_tile_rows, dense_tile_cols);

            double start2 = get_time();


            MMb_mkl(&AH, w, h_out, b);

            double start3 = get_time();

            auto relu_operator = relu<vT>;
            UEwD(h_out, h_out, relu_operator);

            double end = get_time();

            spmm_time += start2 - start1;
            individual_layer_times[layer][0] += start2 - start1;

            mm_time += start3 - start2;
            individual_layer_times[layer][1] += start3 - start2;

            elw_time += end - start3;
            individual_layer_times[layer][2] += end - start3;

            matalloc_time += start1 - start0;
            individual_layer_times[layer][3] += start1 - start0;

            AH.clear();
            break;
        }
    }

    if (refresh_and_print) {

        std::cout << "SpMM took1 : " << spmm_time << std::endl;
        std::cout << "GeMM took1 : " << mm_time << std::endl;
        std::cout << "Matrix Allocation took1 : " << matalloc_time << std::endl;
        std::cout << "Element Wise Unary Dense took1 : " << elw_time << std::endl;
        // print_count();
        spmm_time = 0;
        mm_time = 0;
        matalloc_time = 0;
        elw_time = 0;
    }
}


