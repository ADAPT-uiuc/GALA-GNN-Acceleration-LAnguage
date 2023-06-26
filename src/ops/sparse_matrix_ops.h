//Matrix operations library based on the adopted abstraction.

#include <omp.h>
#include "../matrix/csrc_matrix.h"
#include "../matrix/coo_matrix.h"
#include "../matrix/dense_matrix.h"
#include "../matrix/matrix_prop.h"
#include "../utils/threading_utils.h"

#include "uelw_ops.h"
#include "belw_ops.h"
#include "tiling.h"
#include "aggregators.h"

#ifdef LXSM
#include <libxsmm.h>
#endif

#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

template<class DM, class SM, class Function>
void gSpMM_tiled(const SM *A,
                 const DM *B,
                 DM *out,
                 Function aggregator,
#ifdef LO_KK
        typename DM::ntype k_i,
        typename DM::ntype k_length,
#endif
                 GNOpTile<SM, DM> *tile_info,
                 typename SM::ntype *offset_copy) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
#ifdef GN_1
    vT *A_vals_ptr = A->vals_ptr();
#endif
    nT *A_offset_ptr = offset_copy;
    iT *A_ids_ptr = A->ids_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();

#ifdef LO_I_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
#ifdef LO_KK
        dnT row_offset1 = (dnT) (v * B_cols + k_i);
#else
        dnT row_offset1 = (dnT) v * B_cols;
#endif
        dvT *base1 = out_vals_ptr + row_offset1;
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];

        // iT count_tile = 0;
        // iT tile_diff = tile_info->drows_end - tile_info->drows_start;
        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
            // You can make this more efficient by having an assignable offset array
            if (u >= tile_info->drows_start && u < tile_info->drows_end) {
#ifdef GN_1
                vT A_val = A_vals_ptr[e];
#endif

#ifdef LO_KK
                dnT row_offset2 = (dnT) (u * B_cols + k_i);
#else
                dnT row_offset2 = (dnT) u * B_cols;
#endif
                dvT *base2 = B_vals_ptr + row_offset2;

#ifdef LO_KK
                diT mx_slice = std::min(k_length, B_cols - k_i);
#ifdef GN_1
                aggregator(base1, base2, A_val, mx_slice);
#elif GN_2
                aggregator(base1, base2, mx_slice);
#endif
#else
#ifdef GN_1
                aggregator(base1, base2, A_val, B_cols);
#elif GN_2
                aggregator(base1, base2, B_cols);
#endif
#endif
            } else if (u >= tile_info->drows_end) {
                break;
            }
        }
    }
}

template<class DM, class SM, class Function>
void gSpMM_dgl_tiled(const SM *A,
                     const DM *B,
                     DM *out,
                     Function aggregator,
#ifdef LO_KK
        typename DM::ntype k_i,
        typename DM::ntype k_length,
#endif
                     GNOpTile<SM, DM> *tile_info,
                     typename SM::ntype *offset_copy) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
#ifdef GN_1
    vT *A_vals_ptr = A->vals_ptr();
#endif
    nT *copy_offset_ptr = offset_copy;
    nT *src_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();

#ifdef LO_I_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
#ifdef LO_KK
        dnT row_offset1 = (dnT) (v * B_cols + k_i);
#else
        dnT row_offset1 = (dnT) v * B_cols;
#endif
        dvT *base1 = out_vals_ptr + row_offset1;
        nT first_node_edge = copy_offset_ptr[v];
        nT last_node_edge = src_offset_ptr[v + 1];

        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
            // You can make this more efficient by having an assignable offset array
            if (u < tile_info->drows_end) {
#ifdef GN_1
                vT A_val = A_vals_ptr[e];
#endif
#ifdef LO_KK
                dnT row_offset2 = (dnT) (u * B_cols + k_i);
#else
                dnT row_offset2 = (dnT) u * B_cols;
#endif
                dvT *base2 = B_vals_ptr + row_offset2;

#ifdef LO_KK
                diT mx_slice = std::min(k_length, B_cols - k_i);
#ifdef GN_1
                aggregator(base1, base2, A_val, mx_slice);
#elif GN_2
                aggregator(base1, base2, mx_slice);
#endif
#else
#ifdef GN_1
                aggregator(base1, base2, A_val, B_cols);
#elif GN_2
                aggregator(base1, base2, B_cols);
#endif
#endif

                if (e == last_node_edge - 1) {
                    copy_offset_ptr[v] = last_node_edge;
                }
            } else {
                copy_offset_ptr[v] = e;
                break;
            }
        }
    }
}

template<class DM, class SM, class Function>
void gSpMM_slice_dgl_tiled(const SM *A, const DM *B, DM *out, Function aggregator,
                           GNOpTile<SM, DM> *tile_info,
                           typename SM::ntype *offset_copy,
                           typename DM::itype split_offset,
                           typename DM::itype split_length) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *copy_offset_ptr = offset_copy;
    nT *src_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    dvT *out_vals_ptr = out->vals_ptr() + split_offset;
    dvT *B_vals_ptr = B->vals_ptr() + split_offset;

//    std::cout << "This: " <<  split_offset << " " << split_length << std::endl;

    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
        dnT row_offset1 = (dnT) v * B_cols;
        dvT *base1 = out_vals_ptr + row_offset1;
        nT first_node_edge = copy_offset_ptr[v];
        nT last_node_edge = src_offset_ptr[v + 1];

        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
            // You can make this more efficient by having an assignable offset array
            if (u < tile_info->drows_end) {
                vT A_val = A_vals_ptr[e];
                dnT row_offset2 = (dnT) u * B_cols;
                dvT *base2 = B_vals_ptr + row_offset2;

                aggregator(base1, base2, A_val, split_length);
                if (e == last_node_edge - 1) {
                    copy_offset_ptr[v] = last_node_edge;
                }
            } else {
                copy_offset_ptr[v] = e;
                break;
            }
        }
    }
}


template<class DM, class SM, class Function>
void gSpMM_row_tiled(const SM *A,
                     const DM *B,
                     DM *out,
                     Function aggregator,
#ifdef LO_KK
        typename DM::ntype k_i,
        typename DM::ntype k_length,
#endif
                     GNOpTile<SM, DM> *tile_info) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
#ifdef GN_1
    vT *A_vals_ptr = A->vals_ptr();
#endif
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
#ifdef CCMP
    iT *A_row_ptr = A->row_ids_ptr();
#endif
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();

    auto tempArr = (dvT *) aligned_alloc(64, sizeof(dvT) * B_cols);

#ifdef LO_I_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
#ifdef CCMP
    for (iT v_i = tile_info->srows_start; v_i < tile_info->srows_end; v_i++) {
        auto v = A_row_ptr[v_i];
#else
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
#endif

#ifdef LO_KK
        dnT row_offset1 = (dnT) (v * B_cols + k_i);
#else
        dnT row_offset1 = (dnT) v * B_cols;
#endif
        dvT *base1 = out_vals_ptr + row_offset1;
        std::memset(tempArr, 0, sizeof(dvT) * B_cols);
//        dvT tempArr[B_cols] = {0};
#ifdef CCMP
        nT first_node_edge = A_offset_ptr[v_i];
        nT last_node_edge = A_offset_ptr[v_i + 1];
#else
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];
#endif

        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
#ifdef AU_C
            if (u == -1) {
                continue;
            }
#endif
#ifdef GN_1
            // You can make this more efficient by having an assignable offset array
            vT A_val = A_vals_ptr[e];
#endif

#ifdef LO_KK
            dnT row_offset2 = (dnT) (u * B_cols + k_i);
#else
            dnT row_offset2 = (dnT) u * B_cols;
#endif
            dvT *base2 = B_vals_ptr + row_offset2;


#ifdef LO_KK
            diT mx_slice = std::min(k_length, B_cols - k_i);
#ifdef GN_1
            aggregator(tempArr, base2, A_val, mx_slice);
#elif GN_2
            aggregator(tempArr, base2, mx_slice);
#endif
#else
#ifdef GN_1
            aggregator(tempArr, base2, A_val, B_cols);
//            aggregator(base1, base2, A_val, B_cols);
#elif GN_2
            aggregator(tempArr, base2, B_cols);
//            aggregator(base1, base2, B_cols);
#endif
#endif
        }
        for (dnT k = 0; k < B_cols; k++) {
            base1[k] += tempArr[k];
        }
    }
    free(tempArr);
}

template<class DM, class SM, class Function>
void gSpMM_row_tiled_gn2(const SM *A,
                         const DM *B,
                         DM *out,
                         Function aggregator,
#ifdef LO_KK
        typename DM::ntype k_i,
        typename DM::ntype k_length,
#endif
                         GNOpTile<SM, DM> *tile_info) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
#ifdef CCMP
    iT *A_row_ptr = A->row_ids_ptr();
#endif
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();

    auto tempArr = (dvT *) aligned_alloc(64, sizeof(dvT) * B_cols);

#ifdef LO_I_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
#ifdef CCMP
    for (iT v_i = tile_info->srows_start; v_i < tile_info->srows_end; v_i++) {
        auto v = A_row_ptr[v_i];
#else
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
#endif

#ifdef LO_KK
        dnT row_offset1 = (dnT) (v * B_cols + k_i);
#else
        dnT row_offset1 = (dnT) v * B_cols;
#endif
        dvT *base1 = out_vals_ptr + row_offset1;
        std::memset(tempArr, 0, sizeof(dvT) * B_cols);
//        dvT tempArr[B_cols] = {0};
#ifdef CCMP
        nT first_node_edge = A_offset_ptr[v_i];
        nT last_node_edge = A_offset_ptr[v_i + 1];
#else
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];
#endif

        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
#ifdef AU_C
            if (u == -1) {
                continue;
            }
#endif

#ifdef LO_KK
            dnT row_offset2 = (dnT) (u * B_cols + k_i);
#else
            dnT row_offset2 = (dnT) u * B_cols;
#endif
            dvT *base2 = B_vals_ptr + row_offset2;


#ifdef LO_KK
            diT mx_slice = std::min(k_length, B_cols - k_i);
            aggregator(tempArr, base2, mx_slice);
#else
            aggregator(tempArr, base2, B_cols);
//            aggregator(base1, base2, B_cols);
#endif
        }
        for (dnT k = 0; k < B_cols; k++) {
            base1[k] += tempArr[k];
        }
    }
    free(tempArr);
}

template<class DM, class SM, class Function>
void gSpMM_row_tiled_split(const SM *A, const DM *B, DM *out, Function aggregator,
                           typename DM::ntype k_i,
                           typename DM::ntype k_length,
                           GNOpTile<SM, DM> *tile_info) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
#ifdef CCMP
    iT *A_row_ptr = A->row_ids_ptr();
#endif
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();

#ifdef LO_I_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
#ifdef CCMP
    for (iT v_i = tile_info->srows_start; v_i < tile_info->srows_end; v_i++) {
        auto v = A_row_ptr[v_i];
#else
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
#endif
        dnT row_offset1 = (dnT) (v * B_cols + k_i);
        dvT *base1 = out_vals_ptr + row_offset1;

        // TODO CHECK is adding temp always beneficial??
        //dvT tempArr[B_cols] = {0};
#ifdef CCMP
        nT first_node_edge = A_offset_ptr[v_i];
        nT last_node_edge = A_offset_ptr[v_i + 1];
#else
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];
#endif

        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
#ifdef AU_C
            if (u == -1) {
                continue;
            }
#endif
            // You can make this more efficient by having an assignable offset array
            vT A_val = A_vals_ptr[e];
            dnT row_offset2 = (dnT) (u * B_cols + k_i);
            dvT *base2 = B_vals_ptr + row_offset2;

            //aggregator(tempArr, base2, A_val, B_cols);
            diT mx_slice = std::min(k_length, B_cols - k_i);
            aggregator(base1, base2, A_val, mx_slice);
        }
        // for (dnT k = 0; k < B_cols; k++) {
        //     base1[k] += tempArr[k];
        // }
    }
}

#ifdef AU_C // Remove to add AU_C parts back
template<class DM, class SM, class Function>
void gSpMM_row_tiled_x_weight(const SM *A,
                              const DM *B,
                              const DM *W,
                              const DM *bias,
                              DM *inter,
                              DM *out,
                              Function aggregator,
                              GNOpTile<SM, DM> *tile_info) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    dvT *inter_vals_ptr = inter->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    dvT *W_vals_ptr = W->vals_ptr();
    dvT *bias_vals_ptr = bias->vals_ptr();

    diT I_ncols = inter->ncols();
    diT out_ncols = out->ncols();
    diT out_nrows = out->nrows();
    dvT *out_vals_ptr = out->vals_ptr();

    diT W_ncols = W->ncols();

    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
        dnT row_offset1 = (dnT) v * B_cols;
        dnT out_offset = (dnT) v * W_ncols;
        dvT *base1 = inter_vals_ptr + row_offset1;
        dvT *baseOut = out_vals_ptr + out_offset;
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];

        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
            if (u != -1) {
                // You can make this more efficient by having an assignable offset array
                vT A_val = A_vals_ptr[e];
                dnT row_offset2 = (dnT) u * B_cols;
                dvT *base2 = B_vals_ptr + row_offset2;

                aggregator(base1, base2, A_val, B_cols);
            } else {
                // TODO add this as separate task (pragma task?)
#ifdef GEMM_MKL
                cblas_sgemm(CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            1,
                            W_ncols,
                            I_ncols,
                            1.0f,
                            base1,
                            I_ncols,
                            W_vals_ptr,
                            W_ncols,
                            1.0f,
                            baseOut,
                            W_ncols);
#elif GEMM_OPB
                cblas_sgemm(CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            1,
                            W_ncols,
                            I_ncols,
                            1.0f,
                            base1,
                            I_ncols,
                            W_vals_ptr,
                            W_ncols,
                            1.0f,
                            baseOut,
                            W_ncols);
#endif
            }
        }
    }
}

template<class DM, class SM, class Function>
void gSpMM_row_tiled_t_weight(const SM *A,
                              const DM *B,
                              const DM *W,
                              const DM *bias,
                              DM *inter,
                              DM *out,
                              Function aggregator,
                              GNOpTile<SM, DM> *tile_info) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_ncols = B->ncols();
    iT A_nrows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    dvT *inter_vals_ptr = inter->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    dvT *W_vals_ptr = W->vals_ptr();

    diT inter_ncols = inter->ncols();
    diT out_ncols = out->ncols();
    diT out_nrows = out->nrows();
    dvT *out_vals_ptr = out->vals_ptr();

    diT W_ncols = W->ncols();
    diT W_nrows = W->nrows();

    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
        dnT row_offset1 = (dnT) v * B_ncols;
        dvT *base1 = inter_vals_ptr + row_offset1;
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];

        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
            if (u != -1) {
                // You can make this more efficient by having an assignable offset array
                vT A_val = A_vals_ptr[e];
                dnT row_offset2 = (dnT) u * B_ncols;
                dvT *base2 = B_vals_ptr + row_offset2;
                aggregator(base1, base2, A_val, B_ncols);
            }
        }
    }
#ifdef MM_IMP
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];
        if (first_node_edge != last_node_edge){
            iT u = A_ids_ptr[last_node_edge - 1];

            if (u == -1) {
                dnT inter_offset = (dnT) v * inter_ncols;
                dnT out_offset = (dnT) v * W_ncols;

                if (W->type() == DM::DENSE_MTX_TYPE::CM) {
#pragma omp simd
                    for (diT i_j = 0; i_j < W_ncols; i_j++) {
                        dvT sum = out_vals_ptr[out_offset + i_j];
                        dnT out_col_offset = i_j * W_nrows;
                        for (diT i_k = 0; i_k < inter_ncols; i_k++) {
                            dvT v_i = inter_vals_ptr[(dnT) (inter_offset + i_k)];
                            dvT v_j = W_vals_ptr[(dnT) (i_k + out_col_offset)];
                            sum += v_i * v_j;
                        }
                        out_vals_ptr[out_offset + i_j] = sum;
                    }
                }
            }
        }
    }
#else
#pragma omp parallel for schedule(dynamic, 1)
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];
        if (first_node_edge != last_node_edge) {
            iT u = A_ids_ptr[last_node_edge - 1];

            if (u == -1) {
                dnT row_offset1 = (dnT) v * B_ncols;
                dnT out_offset = (dnT) v * W_ncols;
                dvT *base1 = inter_vals_ptr + row_offset1;
                dvT *baseOut = out_vals_ptr + out_offset;
#ifdef GEMM_MKL
                cblas_sgemm(CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            1,
                            W_ncols,
                            inter_ncols,
                            1.0f,
                            base1,
                            inter_ncols,
                            W_vals_ptr,
                            W_ncols,
                            1.0f,
                            baseOut,
                            W_ncols);
#elif GEMM_OPB
                cblas_sgemm(CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            1,
                            W_ncols,
                            inter_ncols,
                            1.0f,
                            base1,
                            inter_ncols,
                            W_vals_ptr,
                            W_ncols,
                            1.0f,
                            baseOut,
                            W_ncols);
#endif
            }
        }
    }
#endif
}

template<class DM, class SM, class Function>
void gSpMM_row_tiled_p_weight(const SM *A,
                              const DM *B,
                              const DM *W,
                              const DM *bias,
                              DM *inter,
                              DM *out,
                              Function aggregator,
                              GNOpTile<SM, DM> *tile_info) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_ncols = B->ncols();
    iT A_nrows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    dvT *inter_vals_ptr = inter->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    dvT *W_vals_ptr = W->vals_ptr();

    diT inter_ncols = inter->ncols();
    diT out_ncols = out->ncols();
    diT out_nrows = out->nrows();
    dvT *out_vals_ptr = out->vals_ptr();

    bool *A_work = A->work_ptr();

    diT W_ncols = W->ncols();

    // Added this optim
    if (A_offset_ptr[tile_info->srows_start] != A_offset_ptr[tile_info->srows_end + 1]) {
#pragma omp taskgroup
        {
            for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
                dnT row_offset1 = (dnT) v * B_ncols;
                dvT *base1 = inter_vals_ptr + row_offset1;
                nT first_node_edge = A_offset_ptr[v];
                nT last_node_edge = A_offset_ptr[v + 1];

                for (nT e = first_node_edge; e < last_node_edge; e++) {
                    iT u = A_ids_ptr[e];
                    vT A_val = A_vals_ptr[e];
                    dnT row_offset2 = (dnT) u * B_ncols;
                    dvT *base2 = B_vals_ptr + row_offset2;
                    aggregator(base1, base2, A_val, B_ncols);
                }
            }
        }
    }
//#pragma omp taskwait

    iT count_contig = 0;
    iT current_start = -1;
//#pragma omp taskloop
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
        if (A_work[v] == true) {
            if (count_contig <= 0) {
                current_start = v;
            }
            count_contig += 1;
//             if (count_contig == 4){
//                 dnT row_offset1 = (dnT) current_start * B_ncols;
//                 dnT out_offset = (dnT) current_start * W_ncols;
//                 dvT *base1 = inter_vals_ptr + row_offset1;
//                 dvT *baseOut = out_vals_ptr + out_offset;
// #pragma omp task
//                 {
//                     cblas_sgemm(CblasRowMajor,
//                                 CblasNoTrans,
//                                 CblasNoTrans,
//                                 count_contig,
//                                 W_ncols,
//                                 inter_ncols,
//                                 1.0f,
//                                 base1,
//                                 inter_ncols,
//                                 W_vals_ptr,
//                                 W_ncols,
//                                 1.0f,
//                                 baseOut,
//                                 W_ncols);
//                 }
//                 count_contig = 0;
//                 current_start = -1;
//             }
        } else if (count_contig != 0) {
            dnT row_offset1 = (dnT) current_start * B_ncols;
            dnT out_offset = (dnT) current_start * W_ncols;
            dvT *base1 = inter_vals_ptr + row_offset1;
            dvT *baseOut = out_vals_ptr + out_offset;
#ifdef LXSM
            typedef libxsmm_mmfunction <dvT> kernel_type;
            /* generates and dispatches a matrix multiplication kernel (C++ functor) */
            kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, count_contig, W_ncols, inter_ncols, 1.0 /*alpha*/, 1.0 /*beta*/);
            /* kernel multiplies and accumulates matrices: C += Ai * Bi */
            kernel(base1, W_vals_ptr, baseOut);
#else
#pragma omp task
            {
#ifdef GEMM_MKL
                cblas_sgemm(CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            count_contig,
                            W_ncols,
                            inter_ncols,
                            1.0f,
                            base1,
                            inter_ncols,
                            W_vals_ptr,
                            W_ncols,
                            1.0f,
                            baseOut,
                            W_ncols);
#elif GEMM_OPB
                cblas_sgemm(CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            count_contig,
                            W_ncols,
                            inter_ncols,
                            1.0f,
                            base1,
                            inter_ncols,
                            W_vals_ptr,
                            W_ncols,
                            1.0f,
                            baseOut,
                            W_ncols);
#endif
            }
#endif
            count_contig = 0;
            current_start = -1;
        }

    }
    if (count_contig > 0) {
        dnT row_offset1 = (dnT) current_start * B_ncols;
        dnT out_offset = (dnT) current_start * W_ncols;
        dvT *base1 = inter_vals_ptr + row_offset1;
        dvT *baseOut = out_vals_ptr + out_offset;
#ifdef LXSM
        typedef libxsmm_mmfunction <dvT> kernel_type;
        /* generates and dispatches a matrix multiplication kernel (C++ functor) */
        kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, count_contig, W_ncols, inter_ncols, 1.0 /*alpha*/, 1.0 /*beta*/);
        /* kernel multiplies and accumulates matrices: C += Ai * Bi */
        kernel(base1, W_vals_ptr, baseOut);
#else
#pragma omp task
        {
#ifdef GEMM_MKL
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        count_contig,
                        W_ncols,
                        inter_ncols,
                        1.0f,
                        base1,
                        inter_ncols,
                        W_vals_ptr,
                        W_ncols,
                        1.0f,
                        baseOut,
                        W_ncols);
#elif GEMM_OPB
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        count_contig,
                        W_ncols,
                        inter_ncols,
                        1.0f,
                        base1,
                        inter_ncols,
                        W_vals_ptr,
                        W_ncols,
                        1.0f,
                        baseOut,
                        W_ncols);
#endif
        }
#endif
        count_contig = 0;
        current_start = -1;
    }
}

template<class DM, class SM, class Function>
void gSpMM_row_tiled_cs_weight(const SM *A,
                               const DM *B,
                               DM *inter,
                               Function aggregator,
                               GNOpTile<SM, DM> *tile_info) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_ncols = B->ncols();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    dvT *inter_vals_ptr = inter->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();

    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
        dnT row_offset1 = (dnT) v * B_ncols;
        dvT *base1 = inter_vals_ptr + row_offset1;
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];

        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
            vT A_val = A_vals_ptr[e];
            dnT row_offset2 = (dnT) u * B_ncols;
            dvT *base2 = B_vals_ptr + row_offset2;
            aggregator(base1, base2, A_val, B_ncols);
        }
    }
}

template<class DM, class SM>
void gSpMM_row_tiled_cw_weight(const SM *A,
                               const DM *W,
                               DM *inter,
                               DM *out,
                               GNOpTile<SM, DM> *tile_info) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    iT A_nrows = A->nrows();
    dvT *inter_vals_ptr = inter->vals_ptr();
    dvT *W_vals_ptr = W->vals_ptr();

    diT inter_ncols = inter->ncols();
    diT out_ncols = out->ncols();
    diT out_nrows = out->nrows();
    dvT *out_vals_ptr = out->vals_ptr();

    bool *A_work = A->work_ptr();

    diT W_ncols = W->ncols();

    iT count_contig = 0;
    iT current_start = -1;
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
        if (A_work[v] == true) {
            if (count_contig <= 0) {
                current_start = v;
            }
            count_contig += 1;
        } else if (count_contig != 0) {
            dnT row_offset1 = (dnT) current_start * inter_ncols;
            dnT out_offset = (dnT) current_start * W_ncols;
            dvT *base1 = inter_vals_ptr + row_offset1;
            dvT *baseOut = out_vals_ptr + out_offset;
#ifdef GEMM_MKL
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        count_contig,
                        W_ncols,
                        inter_ncols,
                        1.0f,
                        base1,
                        inter_ncols,
                        W_vals_ptr,
                        W_ncols,
                        1.0f,
                        baseOut,
                        W_ncols);
#elif GEMM_OPB
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        count_contig,
                        W_ncols,
                        inter_ncols,
                        1.0f,
                        base1,
                        inter_ncols,
                        W_vals_ptr,
                        W_ncols,
                        1.0f,
                        baseOut,
                        W_ncols);
#endif
            count_contig = 0;
            current_start = -1;
        }
    }
    if (count_contig > 0) {
        dnT row_offset1 = (dnT) current_start * inter_ncols;
        dnT out_offset = (dnT) current_start * W_ncols;
        dvT *base1 = inter_vals_ptr + row_offset1;
        dvT *baseOut = out_vals_ptr + out_offset;
#ifdef GEMM_MKL
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    count_contig,
                    W_ncols,
                    inter_ncols,
                    1.0f,
                    base1,
                    inter_ncols,
                    W_vals_ptr,
                    W_ncols,
                    1.0f,
                    baseOut,
                    W_ncols);
#elif GEMM_OPB
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    count_contig,
                    W_ncols,
                    inter_ncols,
                    1.0f,
                    base1,
                    inter_ncols,
                    W_vals_ptr,
                    W_ncols,
                    1.0f,
                    baseOut,
                    W_ncols);
#endif
    }
}

template<class DM, class SM, class Function>
void gSpMM_row_tiled_p_weight_cts(const SM *A,
                                  const DM *B,
                                  const DM *W,
                                  const DM *bias,
                                  DM *inter,
                                  DM *out,
                                  Function aggregator,
                                  GNOpTile<SM, DM> *tile_info) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_ncols = B->ncols();
    iT A_nrows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    dvT *inter_vals_ptr = inter->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    dvT *W_vals_ptr = W->vals_ptr();

    diT inter_ncols = inter->ncols();
    diT out_ncols = out->ncols();
    diT out_nrows = out->nrows();
    dvT *out_vals_ptr = out->vals_ptr();

    bool *A_work = A->work_ptr();

    diT W_ncols = W->ncols();

    // Added this optim
    if (A_offset_ptr[tile_info->srows_start] != A_offset_ptr[tile_info->srows_end + 1]) {
#pragma omp task
        {
            for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
                dnT row_offset1 = (dnT) v * B_ncols;
                dvT *base1 = inter_vals_ptr + row_offset1;
                nT first_node_edge = A_offset_ptr[v];
                nT last_node_edge = A_offset_ptr[v + 1];

                for (nT e = first_node_edge; e < last_node_edge; e++) {
                    iT u = A_ids_ptr[e];
                    vT A_val = A_vals_ptr[e];
                    dnT row_offset2 = (dnT) u * B_ncols;
                    dvT *base2 = B_vals_ptr + row_offset2;
                    aggregator(base1, base2, A_val, B_ncols);
                }
            }
        }
    }

    iT count_contig = 0;
    iT current_start = -1;
//#pragma omp taskloop
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
        if (A_work[v] == true) {
            if (count_contig <= 0) {
                current_start = v;
            }
            count_contig += 1;
        } else if (count_contig != 0) {
            dnT row_offset1 = (dnT) current_start * B_ncols;
            dnT out_offset = (dnT) current_start * W_ncols;
            dvT *base1 = inter_vals_ptr + row_offset1;
            dvT *baseOut = out_vals_ptr + out_offset;
#ifdef GEMM_MKL
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        count_contig,
                        W_ncols,
                        inter_ncols,
                        1.0f,
                        base1,
                        inter_ncols,
                        W_vals_ptr,
                        W_ncols,
                        1.0f,
                        baseOut,
                        W_ncols);
#elif GEMM_OPB
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        count_contig,
                        W_ncols,
                        inter_ncols,
                        1.0f,
                        base1,
                        inter_ncols,
                        W_vals_ptr,
                        W_ncols,
                        1.0f,
                        baseOut,
                        W_ncols);
#endif
            count_contig = 0;
            current_start = -1;
        }

    }
    if (count_contig > 0) {
        dnT row_offset1 = (dnT) current_start * B_ncols;
        dnT out_offset = (dnT) current_start * W_ncols;
        dvT *base1 = inter_vals_ptr + row_offset1;
        dvT *baseOut = out_vals_ptr + out_offset;
#ifdef GEMM_MKL
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    count_contig,
                    W_ncols,
                    inter_ncols,
                    1.0f,
                    base1,
                    inter_ncols,
                    W_vals_ptr,
                    W_ncols,
                    1.0f,
                    baseOut,
                    W_ncols);
#elif GEMM_OPB
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    count_contig,
                    W_ncols,
                    inter_ncols,
                    1.0f,
                    base1,
                    inter_ncols,
                    W_vals_ptr,
                    W_ncols,
                    1.0f,
                    baseOut,
                    W_ncols);
#endif
        count_contig = 0;
        current_start = -1;
    }
}
#endif // Remove to add the AUC parts back

// TODO check if this is the same as gSpMM_row_tiled_split and remove one
template<class DM, class SM, class Function>
void gSpMM_slice_row_tiled(const SM *A, const DM *B, DM *out, Function aggregator,
                           GNOpTile<SM, DM> *tile_info,
                           typename DM::itype split_offset,
                           typename DM::itype split_length) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
#ifdef CCMP
    iT *A_row_ptr = A->row_ids_ptr();
#endif
    dvT *out_vals_ptr = out->vals_ptr() + split_offset;
    dvT *B_vals_ptr = B->vals_ptr() + split_offset;

#ifdef CCMP
    for (iT v_i = tile_info->srows_start; v_i < tile_info->srows_end; v_i++) {
        auto v = A_row_ptr[v_i];
#else
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
#endif
        dnT row_offset1 = (dnT) v * B_cols;
        dvT *base1 = out_vals_ptr + row_offset1;
#ifdef CCMP
        nT first_node_edge = A_offset_ptr[v_i];
        nT last_node_edge = A_offset_ptr[v_i + 1];
#else
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];
#endif

        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
            // You can make this more efficient by having an assignable offset array
            vT A_val = A_vals_ptr[e];
            dnT row_offset2 = (dnT) u * B_cols;
            dvT *base2 = B_vals_ptr + row_offset2;

            aggregator(base1, base2, A_val, split_length);
        }
    }
}


// void print_count(){
//     std::cout << "50+: " << count50 << " 75+: " << count75 << " 90+: " << count90 << std::endl;
// }


//template<class DM, class SM>
//void SpMM_mkl(const SM * adj, const DM * h_in, DM * AH){
//
//
//    typedef typename SM::itype iT;
//    typedef typename SM::ntype nT;
//    typedef typename SM::vtype vT;
//
//    typedef typename DM::itype diT;
//    typedef typename DM::ntype dnT;
//    typedef typename DM::vtype dvT;
//
//
//
//    iT* ids_ptr = adj->ids_ptr();
//    nT* off_ptr = adj->offset_ptr();
//    nT* off_ptr2 = &adj->offset_ptr()[1];
//
//    const char transa = 'N';
//    const iT A_srows = adj->nrows();
//    const diT B_scols = h_in->ncols();
//    const iT A_scols = adj->ncols();
//    const float alpha_1 = 1.0;
//    const char matdescra[] = {'G', 'L', 'N', 'C'};
//    const float beta_1 = 0.0;
//#ifdef MKL_KERNEL_TEST
//    mkl_scsrmm(&transa, &A_srows, &B_scols, &A_scols, &alpha_1, matdescra, adj->vals_ptr(), ids_ptr,
//            off_ptr, off_ptr2, h_in->vals_ptr(), &B_scols, &beta_1, AH->vals_ptr(), &B_scols);
//#endif
//
//}


template<class DM, class SM, class Function>
void gSpMM(const SM *A, const DM *B, DM *out, Function aggregator) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
//    auto tempArr = (dvT *) aligned_alloc(64, sizeof(dvT) * B_cols);

    if (A->type() == CSRC_TYPE::CSR) {
        nT *A_offset_ptr = A->offset_ptr();
        iT *A_ids_ptr = A->ids_ptr();
#ifndef LO_K_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
        // iterate over all vertices
        for (iT v = 0; v < A_rows; v++) {
            dnT row_offset1 = (dnT) v * B_cols;
            dvT *base1 = out_vals_ptr + row_offset1;

//            dvT tempArr[B_cols] = {0};
//            std::memset(tempArr, 0, B_cols);
            nT first_node_edge = A_offset_ptr[v];
            nT last_node_edge = A_offset_ptr[v + 1];

            // iterate over all edges (adjucent)
            for (nT e = first_node_edge; e < last_node_edge; e++) {
                vT A_val = A_vals_ptr[e];
                iT u = A_ids_ptr[e];
                dnT row_offset2 = (dnT) u * B_cols;
                dvT *base2 = B_vals_ptr + row_offset2;
                //(*aggregator)<dvT, vT, dnT>(base1, base2, A_val, B_cols);
//                aggregator(tempArr, base2, A_val, B_cols);
                aggregator(base1, base2, A_val, B_cols);
            }

//#pragma omp simd
//            for (dnT k = 0; k < B_cols; k++) {
//                base1[k] += tempArr[k];
//            }
        }
    } else if (A->type() == CSRC_TYPE::COO_CO || A->type() == CSRC_TYPE::COO_RO) {
        iT *A_row_ids_ptr = A->row_ids_ptr();
        iT *A_col_ids_ptr = A->col_ids_ptr();
        nT A_vals = A->nvals();

        // iterate over all vertices
        for (nT e = 0; e < A_vals; e++) {
            iT v = A_row_ids_ptr[e];
            iT u = A_col_ids_ptr[e];

            dnT row_offset1 = (dnT) v * B_cols;
            dvT *base1 = out_vals_ptr + row_offset1;
            dnT row_offset2 = (dnT) u * B_cols;
            dvT *base2 = B_vals_ptr + row_offset2;

            vT A_val = A_vals_ptr[e];
            aggregator(base1, base2, A_val, B_cols);
        }
    }
//    free(tempArr);
}

template<class DM, class SM, class Function>
void gSpMM_skip(const SM *A, const DM *B, DM *out, Function aggregator, typename SM::itype skip_v) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();

    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
//    auto tempArr = (dvT *) aligned_alloc(64, sizeof(dvT) * B_cols);
    // iterate over all vertices
    for (iT v = 0; v < A_rows; v++) {
        if (v == skip_v) {
            continue;
        }
        dnT row_offset1 = (dnT) v * B_cols;
        dvT *base1 = out_vals_ptr + row_offset1;

//        dvT tempArr[B_cols] = {0};
//        std::memset(tempArr, 0, sizeof(dvT) * B_cols);
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];

        // iterate over all edges (adjucent)
        for (nT e = first_node_edge; e < last_node_edge; e++) {
            vT A_val = A_vals_ptr[e];
            iT u = A_ids_ptr[e];
            if (u == skip_v) {
                continue;
            }
            dnT row_offset2 = (dnT) u * B_cols;
            dvT *base2 = B_vals_ptr + row_offset2;
//            aggregator(tempArr, base2, A_val, B_cols);
            aggregator(base1, base2, A_val, B_cols);

        }

//#pragma omp simd
//        for (dnT k = 0; k < B_cols; k++) {
//            base1[k] += tempArr[k];
//        }
    }
//    free(tempArr);

}

template<class DM, class SM, class Function>
void gSpMM_set(const SM *A, const DM *B, DM *out, Function aggregator) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
//    auto tempArr = (dvT *) aligned_alloc(64, sizeof(dvT) * B_cols);

    if (A->type() == CSRC_TYPE::CSR) {
        nT *A_offset_ptr = A->offset_ptr();
        iT *A_ids_ptr = A->ids_ptr();
#ifndef LO_K_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
        // iterate over all vertices
        for (iT v = 0; v < A_rows; v++) {
            dnT row_offset1 = (dnT) v * B_cols;
            dvT *base1 = out_vals_ptr + row_offset1;

//            dvT tempArr[B_cols] = {0};
//            std::memset(tempArr, 0, sizeof(dvT) * B_cols);
            nT first_node_edge = A_offset_ptr[v];
            nT last_node_edge = A_offset_ptr[v + 1];

            // iterate over all edges (adjucent)
            for (nT e = first_node_edge; e < last_node_edge; e++) {
                vT A_val = A_vals_ptr[e];
                iT u = A_ids_ptr[e];
                dnT row_offset2 = (dnT) u * B_cols;
                dvT *base2 = B_vals_ptr + row_offset2;
                //(*aggregator)<dvT, vT, dnT>(base1, base2, A_val, B_cols);
//                aggregator(tempArr, base2, A_val, B_cols);
                aggregator(base1, base2, A_val, B_cols);
            }

//#pragma omp simd
//            for (dnT k = 0; k < B_cols; k++) {
//                base1[k] = tempArr[k];
//            }
        }
    } else if (A->type() == CSRC_TYPE::COO_CO || A->type() == CSRC_TYPE::COO_RO) {
        iT *A_row_ids_ptr = A->row_ids_ptr();
        iT *A_col_ids_ptr = A->col_ids_ptr();
        nT A_vals = A->nvals();

        // iterate over all vertices
        for (nT e = 0; e < A_vals; e++) {
            iT v = A_row_ids_ptr[e];
            iT u = A_col_ids_ptr[e];

            dnT row_offset1 = (dnT) v * B_cols;
            dvT *base1 = out_vals_ptr + row_offset1;
            dnT row_offset2 = (dnT) u * B_cols;
            dvT *base2 = B_vals_ptr + row_offset2;

            vT A_val = A_vals_ptr[e];
            aggregator(base1, base2, A_val, B_cols);
        }
    }
//    free(tempArr);
}


template<class DM, class SM, class Function>
void sumSpMM(const SM *A, const DM *B, DM *out, Function aggregator) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
//    auto tempArr = (dvT *) aligned_alloc(64, sizeof(dvT) * B_cols);


#pragma omp parallel for schedule(dynamic, 1)
    // iterate over all vertices
    for (iT v = 0; v < A_rows; v++) {
        dnT row_offset1 = (dnT) v * B_cols;
        dvT *base1 = out_vals_ptr + row_offset1;

//        dvT tempArr[B_cols] = {0};
//        std::memset(tempArr, 0, sizeof(dvT) * B_cols);
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];

        // iterate over all edges (adjucent)
        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
            dnT row_offset2 = (dnT) u * B_cols;
            dvT *base2 = B_vals_ptr + row_offset2;

//            aggregator(tempArr, base2, B_cols);
            aggregator(base1, base2, B_cols);
        }

//#pragma omp simd
//        for (dnT k = 0; k < B_cols; k++) {
//#ifdef INIT_SELF
//            base1[k] += tempArr[k];
//#else
//            base1[k] = tempArr[k];
//#endif
//        }
    }
//    free(tempArr);
}

template<class DM, class SM, class Function>
void sumSpMM_scaleout(const SM *A, const DM *B, DM *out, const DM *S, Function aggregator) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();

    dvT *S_vals_ptr = S->vals_ptr();
//    auto tempArr = (dvT *) aligned_alloc(64, sizeof(dvT) * B_cols);


#pragma omp parallel for schedule(dynamic, 1)
    for (iT v = 0; v < A_rows; v++) {
        dnT row_offset1 = (dnT) v * B_cols;
        dvT *base1 = out_vals_ptr + row_offset1;

//        dvT tempArr[B_cols] = {0};
//        std::memset(tempArr, 0, sizeof(dvT) * B_cols);
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];

        dvT s_val = S_vals_ptr[v];

        // iterate over all edges (adjucent)
        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
            dnT row_offset2 = (dnT) u * B_cols;
            dvT *base2 = B_vals_ptr + row_offset2;

//            aggregator(tempArr, base2, B_cols);
            aggregator(base1, base2, B_cols);
        }

//#pragma omp simd
//        for (dnT k = 0; k < B_cols; k++) {
//            base1[k] = tempArr[k] * s_val;
//        }
    }
//    free(tempArr);
}

template<class DM, class SM>
void fusedGatEdgeKernel(const SM *A, const DM *V1, const DM *V2, SM *out) {
    //Fused GAT kernel for edge computations. It combines gSDDVV, SpMM, and Unary Elementwise on Sparse.
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    iT A_rows = A->nrows();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    vT *out_vals_ptr = out->vals_ptr();
    dvT *V1_vals_ptr = V1->vals_ptr();
    dvT *V2_vals_ptr = V2->vals_ptr();

#pragma omp parallel for
    for (iT v = 0; v < A_rows; v++) {
        vT sum = 0;
        dvT *V1_node = V1_vals_ptr + (dnT) v;
        for (nT e = A_offset_ptr[v]; e < A_offset_ptr[v + 1]; e++) {
            iT u = A_ids_ptr[e];
            vT temp = exp(lrelu<vT>(*V1_node + V2_vals_ptr[(dnT) u]));
            out_vals_ptr[e] = temp;
            sum += temp;
        }
        // Apply softmax
        for (nT e = A_offset_ptr[v]; e < A_offset_ptr[v + 1]; e++) {
            out_vals_ptr[e] = out_vals_ptr[e] / sum;
        }
    }
}


template<class DM, class SM, class Function>
void gSpMV(const SM *A, const DM *V, DM *out, Function aggregator) {
    //Generalized SpMV.
    //The aggregator here is an operator between the accumulator and adj nodes scalar.
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *V_vals_ptr = V->vals_ptr();


#pragma omp parallel for
    for (iT v = 0; v < A_rows; v++) {
        //TODO:The initial value of the accumulator might need to change if zero is not neutral element for the aggregator.
        vT accum = 0;
        dvT *V_node = V_vals_ptr + (dnT) v;
        for (nT e = A_offset_ptr[v]; e < A_offset_ptr[v + 1]; e++) {
            accum = aggregator(accum, A_vals_ptr[e], *V_node);
        }
        out_vals_ptr[(dnT) v] = accum;
    }
}

template<class DM, class SM>
void SpMV_ones(const SM *A, DM *out) {
    //SpMV with the vector being ones.
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();

#ifdef SM_1
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();

#pragma omp parallel for schedule(dynamic, 1)
    for (iT v = 0; v < A_rows; v++) {
        vT accum = 0;
        for (nT e = A_offset_ptr[v]; e < A_offset_ptr[v + 1]; e++) {
            accum = accum + A_vals_ptr[e];;
        }
        out_vals_ptr[(dnT) v] = accum;
    }
#endif
#ifdef SM_2
    iT *A_row_ids_ptr = A->row_ids_ptr();
    nT A_nvals = A->nvals();

#pragma omp parallel for schedule(dynamic, 1)
    for (nT e = 0; e < A_nvals; e++) {
        out_vals_ptr[A_row_ids_ptr[e]] += A_vals_ptr[e];;
    }
#endif
}

template<class DM, class SM>
void SpMV_ones_vec(const std::vector<SM *> A_vec, DM *out) {
    //SpMV with the vector being ones.
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dvT *out_vals_ptr = out->vals_ptr();

    for (iT j = 0; j < A_vec.size(); j += 1) {
        SM *A = A_vec.at(j);
        iT A_rows = A->nrows();
        vT *A_vals_ptr = A->vals_ptr();
        iT *A_row_ptr = A->row_ids_ptr();
        nT *A_offset_ptr = A->offset_ptr();
        iT *A_ids_ptr = A->ids_ptr();

#pragma omp parallel for schedule(dynamic, 8)
        for (iT v_i = 0; v_i < A_rows; v_i++) {
            auto v = A_row_ptr[v_i];

            nT first_node_edge = A_offset_ptr[v_i];
            nT last_node_edge = A_offset_ptr[v_i + 1];
            vT accum = 0;
            for (nT e = first_node_edge; e < last_node_edge; e++) {
                accum = accum + A_vals_ptr[e];;
            }
            out_vals_ptr[(dnT) v] += accum;
        }
    }
}

template<class DM, class SM, class Function>
void SpVRBM(const SM *A, const DM *V, SM *out, Function boperator) {
    // Sparse matrix vector row broadcast multiplication
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    //iT * A_ids_ptr = A->ids_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *V_vals_ptr = V->vals_ptr();


#pragma omp parallel for
    for (iT v = 0; v < A_rows; v++) {
        dvT *V_node = V_vals_ptr + (dnT) v;
#pragma omp simd
        for (nT e = A_offset_ptr[v]; e < A_offset_ptr[v + 1]; e++) {
            out_vals_ptr[e] = boperator(A_vals_ptr[e], *V_node);
        }
    }
}

template<class DM, class SM, class Function>
void SpVCBM(const SM *A, const DM *V, SM *out, Function boperator) {
    // Sparse matrix vector column broadcast multiplication
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *V_vals_ptr = V->vals_ptr();


#pragma omp parallel for
    for (iT v = 0; v < A_rows; v++) {
#pragma omp simd
        for (nT e = A_offset_ptr[v]; e < A_offset_ptr[v + 1]; e++) {
            iT u = A_ids_ptr[e];
            dvT *V_node = V_vals_ptr + (dnT) u;
            out_vals_ptr[e] = boperator(A_vals_ptr[e], *V_node);
        }
    }
}

template<class DM, class Function>
void DVCBM(const DM *V, DM *out, Function boperator) {
    // Inplace Dence matrix vector column broadcast multiplication
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    diT out_rows = out->nrows();
    diT out_cols = out->ncols();
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *V_vals_ptr = V->vals_ptr();

#pragma omp parallel for
    for (diT i_i = 0; i_i < out_rows; i_i++) {
        dnT out_row_offset = i_i * out_cols;
#pragma omp simd
        for (diT i_j = 0; i_j < out_cols; i_j++) {
            out_vals_ptr[out_row_offset + i_j] = boperator(out_vals_ptr[out_row_offset + i_j], V_vals_ptr[i_j]);
        }
    }
}

template<class SM, class Function>
void UEwS(const SM *in, SM *out, Function uoperator) {
    //Unary Elementwise on Sparse.
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;


    iT in_rows = in->nrows();
    vT *in_vals_ptr = in->vals_ptr();
    nT *in_offset_ptr = in->offset_ptr();
    vT *out_vals_ptr = out->vals_ptr();


#pragma omp parallel for schedule(dynamic, 1)
    for (iT v = 0; v < in_rows; v++) {
        for (nT e = in_offset_ptr[v]; e < in_offset_ptr[v + 1]; e++) {
            out_vals_ptr[e] = uoperator(in_vals_ptr[e]);
        }
    }
}

template<class DM, class Function>
void UEwD(const DM *in, DM *out, Function uoperator) {
    //Unary Elementwise on Sparse.
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;


    diT in_rows = in->nrows();
    diT in_cols = in->ncols();
    dvT *in_vals_ptr = in->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();


#pragma omp parallel for schedule(static)
    for (diT i_i = 0; i_i < in_rows; i_i++) {
        dnT in_row_offset = i_i * in_cols;
        for (diT i_j = 0; i_j < in_cols; i_j++) {
            out_vals_ptr[in_row_offset + i_j] = uoperator(in_vals_ptr[in_row_offset + i_j]);
        }
    }
}

template<class SM, class DM, class Function>
void gSDDVV(const SM *A,
            const DM *V1,
            const DM *V2,
            SM *out,
            Function boperator) {
    //Generalized SDDVV.
    //The aggregator here is an operator between the accumulator and adj nodes scalar.
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    iT A_rows = A->nrows();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    vT *out_vals_ptr = out->vals_ptr();
    dvT *V1_vals_ptr = V1->vals_ptr();
    dvT *V2_vals_ptr = V2->vals_ptr();

#pragma omp parallel for
    for (iT v = 0; v < A_rows; v++) {
        dvT V1_val = V1_vals_ptr[(dnT) v];
        for (nT e = A_offset_ptr[v]; e < A_offset_ptr[v + 1]; e++) {
            iT u = A_ids_ptr[e];
            dvT V2_val = V2_vals_ptr[(dnT) u];
//            out_vals_ptr[e] = boperator(*V1_node, V2_vals_ptr[(dnT) u]);
            out_vals_ptr[e] = boperator(V1_val, V2_val);
            //out_vals_ptr[e] = *V1_node + V2_vals_ptr[(dnT)u];
        }
    }
}

template<class SM, class DM, class Function>
void gSDDMM(const SM *A, const DM *M1, const DM *M2, SM *out, Function roperator) {
    //Generalized SDDMM
    //The aggregator here is an operator between the accumulator and adj nodes scalar.
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    vT *out_vals_ptr = out->vals_ptr();
    dvT *M1_vals_ptr = M1->vals_ptr();
    dvT *M2_vals_ptr = M2->vals_ptr();
    diT k = M1->ncols();

#pragma omp parallel for schedule(dynamic, 8)
    // iterate over all vertices
    for (iT v = 0; v < A_rows; v++) {
        dnT row_offset1 = (dnT) v * k;
        dvT *base1 = M1_vals_ptr + row_offset1;
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];

        // TODO check performance with an added parallel here
        // iterate over all edges (adjucent)
        for (nT e = first_node_edge; e < last_node_edge; e++) {
            vT A_val = A_vals_ptr[e];
            iT u = A_ids_ptr[e];
            dnT row_offset2 = (dnT) u * k;
            dvT *base2 = M1_vals_ptr + row_offset2;
            //(*aggregator)<dvT, vT, dnT>(base1, base2, A_val, B_cols);
            A_vals_ptr[e] = roperator(base1, base2, A_val, k);
        }
    }
}


template<class SM, class DM, class BOP, class UOP>
void gSDDVV(const SM *A,
            const DM *V1,
            const DM *V2,
            SM *out,
            BOP boperator,
            UOP uoperator,
            GNOpTile<SM, DM> *tile_info) {
    //Generalized SDDVV.
    //The aggregator here is an operator between the accumulator and adj nodes scalar.
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
#ifdef CCMP
    iT *A_row_ptr = A->row_ids_ptr();
#endif
    vT *out_vals_ptr = out->vals_ptr();
    dvT *V1_vals_ptr = V1->vals_ptr();
    dvT *V2_vals_ptr = V2->vals_ptr();

#ifdef LO_I_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
#ifdef CCMP
    for (iT v_i = tile_info->srows_start; v_i < tile_info->srows_end; v_i++) {
        auto v = A_row_ptr[v_i];
#else
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
#endif
#ifdef CCMP
        nT first_node_edge = A_offset_ptr[v_i];
        nT last_node_edge = A_offset_ptr[v_i + 1];
#else
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];
#endif
        dvT V1_val = V1_vals_ptr[(dnT) v];
        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
            dvT V2_val = V2_vals_ptr[(dnT) u];
            out_vals_ptr[e] = uoperator(boperator(V1_val, V2_val));
        }
    }
}

/// ********************************************************************************************************************
/// Transfer learning
/// ********************************************************************************************************************
template<class SM, class DM, class Function>
void trans_jj_iip_i_j_kv(std::vector<SM *> adj_vec,
                         DM *inp_dense,
                         DM *out_dense,
                         typename SM::itype sparse_tile_rows,
                         Function wsum_aggr
) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    for (diT j = 0; j < adj_vec.size(); j += 1) {
#pragma omp parallel for schedule(dynamic, 1)
        for (iT i = 0; i < adj_vec.at(j)->nrows(); i += sparse_tile_rows) {
            GNOpTile<SM, DM> tile_info;

            tile_info.srows_start = i;
            tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j)->nrows());

            gSpMM_row_transf(adj_vec.at(j),
                             inp_dense,
                             out_dense,
                             wsum_aggr,
                             &tile_info);
        }
    }
}

template<class SM, class DM, class Function>
void bless_jj_iip_i_j_kv(std::vector<SM *> &adj_vec,
                         DM *inp_dense,
                         DM *out_dense,
                         typename SM::itype sparse_tile_rows,
                         Function wsum_aggr
) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

#ifdef CCMP
    std::cout << "Barrier-less won't work with DCSR, switch to CSR (set C_COMP to OFF)" << std::endl;
#else
#pragma omp parallel
#pragma omp single nowait
    {
        for (iT i = 0; i < adj_vec.at(0)->nrows(); i += sparse_tile_rows) {
            auto tile_info = new GNOpTile<SM, DM>();

            tile_info->srows_start = i;
            tile_info->srows_end = std::min(i + sparse_tile_rows, adj_vec.at(0)->nrows());
//            std::cout << "Row: " << i + sparse_tile_rows << " "  << adj_vec.at(0)->nrows() << std::endl;

#pragma omp task
            {
                gSpMM_row_bless(adj_vec,
                                0,
                                inp_dense,
                                out_dense,
                                wsum_aggr,
                                tile_info);
            }
        }
    }

//#pragma omp parallel
//    for (iT i = 0; i < adj_vec.at(j)->nrows(); i += sparse_tile_rows) {
//        GNOpTile<SM, DM> tile_info;
//
//        tile_info.srows_start = i;
//        tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j)->nrows());
//
//#pragma omp task
//        {
//            gSpMM_row_bless(adj_vec,
//                            j,
//                            inp_dense,
//                            out_dense,
//                            wsum_aggr,
//                            &tile_info);
//        };
//    }

#endif
}

template<class SM, class DM, class Function>
void trans_kk_jj_iip_i_j_kv(std::vector<SM *> &adj_vec,
                            DM *inp_dense,
                            DM *out_dense,
                            typename SM::itype sparse_tile_rows,
                            Function wsum_aggr,
                            typename DM::itype dense_tile_cols
) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    for (diT k = 0; k < inp_dense->ncols(); k += dense_tile_cols) {
        for (diT j = 0; j < adj_vec.size(); j += 1) {
#pragma omp parallel for schedule(dynamic, 1)
            for (iT i = 0; i < adj_vec.at(j)->nrows(); i += sparse_tile_rows) {
                GNOpTile<SM, DM> tile_info;

                tile_info.srows_start = i;
                tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j)->nrows());

                gSpMM_row_transf_kk(adj_vec.at(j),
                                    inp_dense,
                                    out_dense,
                                    wsum_aggr,
                                    k,
                                    dense_tile_cols,
                                    &tile_info);
            }
        }
    }
}

template<class SM, class DM, class Function>
void slice_kk_jj_iip_i_j_kv(std::vector<SM *> &adj_vec,
                            std::vector<DM *> &inp_dense_vec,
                            std::vector<DM *> &out_dense_vec,
                            typename SM::itype sparse_tile_rows,
                            Function wsum_aggr
) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    for (diT k = 0; k < inp_dense_vec.size(); k += 1) {
        for (diT j = 0; j < adj_vec.size(); j += 1) {
#pragma omp parallel for schedule(dynamic, 1)
            for (iT i = 0; i < adj_vec.at(j)->nrows(); i += sparse_tile_rows) {
                GNOpTile<SM, DM> tile_info;

                tile_info.srows_start = i;
                tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j)->nrows());

                gSpMM_row_transf(adj_vec.at(j),
                                 inp_dense_vec.at(k),
                                 out_dense_vec.at(k),
                                 wsum_aggr,
                                 &tile_info);
            }
        }
//        std::cout << inp_dense_vec.at(k)->vals_ptr()[0] << std::endl;
    }
}

template<class SM, class DM, class Function>
void trans_jj_kk_iip_i_j_kv(std::vector<SM *> &adj_vec,
                            DM *inp_dense,
                            DM *out_dense,
                            typename SM::itype sparse_tile_rows,
                            Function wsum_aggr,
                            typename DM::itype dense_tile_cols
) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    for (diT j = 0; j < adj_vec.size(); j += 1) {
        for (diT k = 0; k < inp_dense->ncols(); k += dense_tile_cols) {
#pragma omp parallel for schedule(dynamic, 1)
            for (iT i = 0; i < adj_vec.at(j)->nrows(); i += sparse_tile_rows) {
                GNOpTile<SM, DM> tile_info;

                tile_info.srows_start = i;
                tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j)->nrows());

                gSpMM_row_transf_kk(adj_vec.at(j),
                                    inp_dense,
                                    out_dense,
                                    wsum_aggr,
                                    k,
                                    dense_tile_cols,
                                    &tile_info);
            }
        }
    }
}

template<class DM, class SM, class Function>
void gSpMM_row_transf(const SM *A,
                      const DM *B,
                      DM *out,
                      Function aggregator,
                      GNOpTile<SM, DM> *tile_info) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
#ifdef CCMP
    iT *A_row_ptr = A->row_ids_ptr();
#endif
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();

    auto tempArr = (dvT *) aligned_alloc(64, sizeof(dvT) * B_cols);

#ifdef CCMP
    for (iT v_i = tile_info->srows_start; v_i < tile_info->srows_end; v_i++) {
        auto v = A_row_ptr[v_i];
#else
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
#endif

        dnT row_offset1 = (dnT) v * B_cols;
        dvT *base1 = out_vals_ptr + row_offset1;
        std::memset(tempArr, 0, sizeof(dvT) * B_cols);
#ifdef CCMP
        nT first_node_edge = A_offset_ptr[v_i];
        nT last_node_edge = A_offset_ptr[v_i + 1];
#else
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];
#endif

        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
#ifdef GN_1
            // You can make this more efficient by having an assignable offset array
            vT A_val = A_vals_ptr[e];
#endif
            dnT row_offset2 = (dnT) u * B_cols;
            dvT *base2 = B_vals_ptr + row_offset2;


            aggregator(tempArr, base2, A_val, B_cols);
//            aggregator(base1, base2, A_val, B_cols);
        }
        for (dnT k = 0; k < B_cols; k++) {
            base1[k] += tempArr[k];
        }
    }
    free(tempArr);
}

template<class DM, class SM, class Function>
void gSpMM_row_bless(std::vector<SM *> &adj_vec,
                     typename DM::itype jj,
                     const DM *B,
                     DM *out,
                     Function aggregator,
                     GNOpTile<SM, DM> *tile_info) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    SM *A = adj_vec.at(jj);

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();

//    auto tempArr = (dvT *) aligned_alloc(64, sizeof(dvT) * B_cols);

    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {

        dnT row_offset1 = (dnT) v * B_cols;
        dvT *base1 = out_vals_ptr + row_offset1;
//        std::memset(tempArr, 0, sizeof(dvT) * B_cols);
//        std::cout << v << std::endl;
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];

        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
//            std::cout << "u: " << u << "," << v << " " << e << " " << jj << "|" << tile_info->srows_start << " " << tile_info->srows_end << std::endl;
#ifdef GN_1
            // You can make this more efficient by having an assignable offset array
            vT A_val = A_vals_ptr[e];
#endif
            dnT row_offset2 = (dnT) u * B_cols;
            dvT *base2 = B_vals_ptr + row_offset2;


//            aggregator(tempArr, base2, A_val, B_cols);
            aggregator(base1, base2, A_val, B_cols);
        }
//        for (dnT k = 0; k < B_cols; k++) {
//            base1[k] += tempArr[k];
//        }
    }
//    free(tempArr);

//    std::cout << "this is called" << adj_vec.size() << " " << jj + 1 << std::endl;
    if (adj_vec.size() > (jj + 1)) {
//        std::cout << "this is called" << std::endl;
#pragma omp task
        {
//            std::cout << "Call: " << tile_info->srows_start << " " << tile_info->srows_end << std::endl;
            gSpMM_row_bless(adj_vec,
                            jj + 1,
                            B,
                            out,
                            aggregator,
                            tile_info);
        }
    } else {
        delete tile_info;
    }
}

template<class DM, class SM, class Function>
void gSpMM_row_transf_kk(const SM *A,
                         const DM *B,
                         DM *out,
                         Function aggregator,
                         typename DM::ntype k_i,
                         typename DM::ntype k_length,
                         GNOpTile<SM, DM> *tile_info) {
    //Generalized SpMM.
    //The output dense matrix line is used as an accumulator for the aggregated neighbour vectors.

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dnT B_cols = B->ncols();
    iT A_rows = A->nrows();
    vT *A_vals_ptr = A->vals_ptr();
    nT *A_offset_ptr = A->offset_ptr();
    iT *A_ids_ptr = A->ids_ptr();
#ifdef CCMP
    iT *A_row_ptr = A->row_ids_ptr();
#endif
    dvT *out_vals_ptr = out->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    diT mx_slice = std::min(k_length, B_cols - k_i);

    auto tempArr = (dvT *) aligned_alloc(64, sizeof(dvT) * B_cols);

#ifdef CCMP
    for (iT v_i = tile_info->srows_start; v_i < tile_info->srows_end; v_i++) {
        auto v = A_row_ptr[v_i];
#else
    for (iT v = tile_info->srows_start; v < tile_info->srows_end; v++) {
#endif

        dnT row_offset1 = (dnT) (v * B_cols + k_i);
        dvT *base1 = out_vals_ptr + row_offset1;
        std::memset(tempArr, 0, sizeof(dvT) * B_cols);
#ifdef CCMP
        nT first_node_edge = A_offset_ptr[v_i];
        nT last_node_edge = A_offset_ptr[v_i + 1];
#else
        nT first_node_edge = A_offset_ptr[v];
        nT last_node_edge = A_offset_ptr[v + 1];
#endif

        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = A_ids_ptr[e];
            // You can make this more efficient by having an assignable offset array
            vT A_val = A_vals_ptr[e];
            dnT row_offset2 = (dnT) (u * B_cols + k_i);
            dvT *base2 = B_vals_ptr + row_offset2;
            aggregator(tempArr, base2, A_val, mx_slice);
        }
        for (dnT k = 0; k < B_cols; k++) {
            base1[k] += tempArr[k];
        }
    }
    free(tempArr);
}

/// ********************************************************************************************************************
/// Tiled executions
/// ********************************************************************************************************************
template<class SM, class DM, class Function>
void tile_jj_ii_i_j_kv(std::vector<SM *> adj_vec,
                       DM *inp_dense,
                       DM *out_dense,
                       typename SM::itype sparse_tile_rows,
                       Function wsum_aggr
#if defined(ST_1) || defined(ST_2)
        , typename SM::itype sparse_tile_cols
#endif
#ifdef LO_KK
        , typename DM::itype dense_tile_cols
#endif
) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

//#ifdef LO_II_P
////#pragma omp parallel for schedule(guided)
////#pragma omp parallel for schedule(dynamic, 32)
//#pragma omp parallel
//    {
//#endif

#ifdef ST_0

#ifdef LO_KK_JJ_II
#ifdef LO_KK_P
#pragma omp parallel for schedule(static)
#endif
    for (diT k = 0; k < inp_dense->ncols(); k += dense_tile_cols) {
#endif
    for (diT j = 0; j < adj_vec.size(); j += 1) {
#ifdef LO_JJ_KK_II
#ifdef LO_KK_P
#pragma omp parallel for schedule(static)
#endif
        for (diT k = 0; k < inp_dense->ncols(); k += dense_tile_cols) {
#endif
#ifdef LO_II_P
        //#pragma omp for nowait schedule(dynamic, 4)
        //#pragma omp parallel for schedule(guided)
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (iT i = 0; i < adj_vec.at(j)->nrows(); i += sparse_tile_rows) {
            GNOpTile<SM, DM> tile_info;

            tile_info.srows_start = i;
            tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j)->nrows());

            gSpMM_row_tiled(adj_vec.at(j),
                            inp_dense,
                            out_dense,
                            wsum_aggr,
#ifdef LO_KK
                    k,
                    dense_tile_cols,
#endif
                            &tile_info);
        }
#ifdef LO_JJ_KK_II
        }
#endif
//#ifdef LO_II_P
//        }
//#endif
    }
#ifdef LO_KK_JJ_II
    }
#endif
#elif  defined(ST_1)
    SM *adj = adj_vec.at(0);
#ifdef LO_KK_JJ_II
#ifdef LO_KK_P
#pragma omp parallel for schedule(static)
#endif
        for (diT k = 0; k < inp_dense->ncols(); k += dense_tile_cols) {
#endif
    for (diT j = 0; j < adj->ncols(); j += sparse_tile_cols) {
        diT drows_end = std::min(j + sparse_tile_cols, adj->ncols());
#ifdef LO_JJ_KK_II
#ifdef LO_KK_P
#pragma omp parallel for schedule(static)
#endif
        for (diT k = 0; k < inp_dense->ncols(); k += dense_tile_cols) {
#endif
#ifdef LO_II_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (iT i = 0; i < adj->nrows(); i += sparse_tile_rows) {
                GNOpTile<SM, DM> tile_info;

                tile_info.srows_start = i;
                tile_info.srows_end = std::min(i + sparse_tile_rows, adj->nrows());
                tile_info.drows_start = j;
                tile_info.drows_end = drows_end;

                gSpMM_tiled(adj,
                            inp_dense,
                            out_dense,
                            wsum_aggr,
#ifdef LO_KK
                            k,
                            dense_tile_cols,
#endif
                            &tile_info,
                            adj->offset_ptr());
            }
#ifdef LO_JJ_KK_II
        }
#endif
    }
#ifdef LO_KK_JJ_II
            }
#endif
#elif ST_2
    SM *adj = adj_vec.at(0);
    nT *copy_offsets = (nT *) aligned_alloc(64, (adj->nrows() + 1) * sizeof(nT));
#ifdef LO_KK
    nT *copy_offsets2 = (nT *) aligned_alloc(64, (adj->nrows() + 1) * sizeof(nT));
#endif
    auto offset_ptr = adj->offset_ptr();

//#pragma omp parallel for simd schedule(static)
//    for (iT c_i = 0; c_i < adj->nrows(); c_i++) {
//        copy_offsets[c_i] = offset_ptr[c_i];
//    }
    memcpy(copy_offsets, adj->offset_ptr(), (adj->nrows() + 1) * sizeof(nT));

#ifdef LO_KK_JJ_II
#ifdef LO_KK_P
#pragma omp parallel for schedule(static)
#endif
        for (diT k = 0; k < inp_dense->ncols(); k += dense_tile_cols) {
//#pragma omp parallel for simd schedule(static)
//            for (iT c_i = 0; c_i < adj->nrows(); c_i++) {
//                copy_offsets2[c_i] = copy_offsets[c_i];
//            }
            memcpy(copy_offsets2, copy_offsets, (adj->nrows() + 1) * sizeof(nT));
#endif
    for (diT j = 0; j < adj->ncols(); j += sparse_tile_cols) {

        diT drows_end = std::min(j + sparse_tile_cols, adj->ncols());
#ifdef LO_JJ_KK_II
#ifdef LO_KK_P
#pragma omp parallel for schedule(static)
#endif
        for (diT k = 0; k < inp_dense->ncols(); k += dense_tile_cols) {
//#pragma omp parallel for simd schedule(static)
//            for (iT c_i = 0; c_i < adj->nrows(); c_i++) {
//                copy_offsets2[c_i] = copy_offsets[c_i];
//            }
            memcpy(copy_offsets2, copy_offsets, (adj->nrows() + 1) * sizeof(nT));
#endif

#ifdef LO_II_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (iT i = 0; i < adj->nrows(); i += sparse_tile_rows) {
                GNOpTile<SM, DM> tile_info;

                tile_info.srows_start = i;
                tile_info.srows_end = std::min(i + sparse_tile_rows, adj->nrows());
                tile_info.drows_start = j;
                tile_info.drows_end = drows_end;

                gSpMM_dgl_tiled(adj,
                                inp_dense,
                                out_dense,
                                wsum_aggr,
#ifdef LO_KK
                                k,
                                dense_tile_cols,
#endif
                                &tile_info,
#ifdef LO_KK
                                copy_offsets2
#else
                                copy_offsets
#endif
                                );
            }
#ifdef LO_JJ_KK_II
        }
//#pragma omp parallel for simd schedule(static)
//        for (iT c_i = 0; c_i < adj->nrows(); c_i++) {
//            copy_offsets[c_i] = copy_offsets2[c_i];
//        }
        memcpy(copy_offsets, copy_offsets2, (adj->nrows() + 1) * sizeof(nT));
#endif
    }
#ifdef LO_KK_JJ_II
            }
#endif
    free(copy_offsets);
#ifdef LO_KK
    free(copy_offsets2);
#endif
#endif
}

template<class SM, class DM>
void tile_jj_iip_i_j_kv_gn2(std::vector<SM *> adj_vec,
                            DM *inp_dense,
                            DM *out_dense,
                            typename SM::itype sparse_tile_rows
#if defined(ST_1) || defined(ST_2)
        , typename SM::itype sparse_tile_cols
#endif
#ifdef LO_KK
        , typename DM::itype dense_tile_cols
#endif
) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    auto wsum_aggr = sumAgg<dvT, dnT>;

#ifdef LO_KK_JJ_II
#ifdef LO_KK_P
#pragma omp parallel for schedule(static)
#endif
    for (diT k = 0; k < inp_dense->ncols(); k += dense_tile_cols) {
#endif
    for (diT j = 0; j < adj_vec.size(); j += 1) {
#ifdef LO_JJ_KK_II
#ifdef LO_KK_P
#pragma omp parallel for schedule(static)
#endif
        for (diT k = 0; k < inp_dense->ncols(); k += dense_tile_cols) {
#endif
#ifdef LO_II_P
        //#pragma omp for nowait schedule(dynamic, 4)
        //#pragma omp parallel for schedule(guided)
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (iT i = 0; i < adj_vec.at(j)->nrows(); i += sparse_tile_rows) {
            GNOpTile<SM, DM> tile_info;

            tile_info.srows_start = i;
            tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j)->nrows());

            gSpMM_row_tiled_gn2(adj_vec.at(j),
                                inp_dense,
                                out_dense,
                                wsum_aggr,
#ifdef LO_KK
                    k,
                    dense_tile_cols,
#endif
                                &tile_info);
        }
#ifdef LO_JJ_KK_II
        }
#endif
//#ifdef LO_II_P
//        }
//#endif
    }
#ifdef LO_KK_JJ_II
    }
#endif
}

// TODO have a better way of doing this than copying the function wholesale
template<class SM, class DM>
void tile_jj_iip_i_j_kv_sddvv(std::vector<SM *> adj_vec_inp,
                              DM *inp_dense,
                              DM *out_dense,
                              std::vector<SM *> adj_vec_out,
                              typename SM::itype sparse_tile_rows) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    auto sum_operator = sum<vT>;
    auto explrelu_operator = explrelu<vT>;
    for (diT j = 0; j < adj_vec_inp.size(); j += 1) {
#ifdef LO_II_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (iT i = 0; i < adj_vec_inp.at(j)->nrows(); i += sparse_tile_rows) {
            GNOpTile<SM, DM> tile_info;

            tile_info.srows_start = i;
            tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec_inp.at(j)->nrows());

            gSDDVV(adj_vec_inp.at(j),
                   inp_dense,
                   out_dense,
                   adj_vec_out.at(j),
                   sum_operator,
                   explrelu_operator,
                   &tile_info);
        }
    }
}

template<class SM, class DM>
void tile_jj_iip_i_j_kv_sddvv(std::vector<SM *> adj_vec_inp,
                              DM *V1_dense,
                              DM *V2_dense,
                              SM *adj_out,
                              typename SM::itype sparse_tile_rows) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    auto sum_operator = sum<vT>;
    auto explrelu_operator = explrelu<vT>;
    for (diT j = 0; j < adj_vec_inp.size(); j += 1) {
#ifdef LO_II_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (iT i = 0; i < adj_vec_inp.at(j)->nrows(); i += sparse_tile_rows) {
            GNOpTile<SM, DM> tile_info;

            tile_info.srows_start = i;
            tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec_inp.at(j)->nrows());

            gSDDVV(adj_vec_inp.at(j),
                   V1_dense,
                   V2_dense,
                   adj_out,
                   sum_operator,
                   explrelu_operator,
                   &tile_info);
        }
    }
}

template<class SM, class DM>
void variable_tile_jj_iip_i_j_kv(std::vector<SM *> adj_vec,
                                 DM *inp_dense,
                                 DM *out_dense,
                                 std::vector<std::vector<GNOpTile<SM, DM>>> &tile_infos) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    auto wsum_aggr = wsumAgg<dvT, vT, dnT>;

    for (diT j = 0; j < adj_vec.size(); j += 1) {
        auto row_tile_vec = tile_infos.at(j);
#ifdef LO_II_P
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (iT t_i = 0; t_i < row_tile_vec.size(); t_i += 1) {
            gSpMM_row_tiled(adj_vec.at(j),
                            inp_dense,
                            out_dense,
                            wsum_aggr,
                            &row_tile_vec.at(t_i));
        }
    }
}

//template<class SM, class DM>
//void tile_jj_kk_ii_i_j_k(std::vector<SM *> adj_vec,
//                         DM *inp_dense,
//                         DM *out_dense,
//                         typename SM::itype sparse_tile_rows,
//                         typename DM::itype dense_tile_cols) {
//    typedef typename SM::itype iT;
//    typedef typename SM::ntype nT;
//    typedef typename SM::vtype vT;
//
//    typedef typename DM::itype diT;
//    typedef typename DM::ntype dnT;
//    typedef typename DM::vtype dvT;
//
//    auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
//    for (diT j = 0; j < adj_vec.size(); j += 1) {
//#ifdef LO_KK_P
//#pragma omp parallel for schedule(static)
//#endif
//        for (diT k = 0; k < inp_dense->ncols(); k += dense_tile_cols) {
////            std::cout << k << std::endl;
//#ifdef LO_II_P
//#pragma omp parallel for schedule(dynamic, 1)
//#endif
//            for (iT i = 0; i < adj_vec.at(j)->nrows(); i += sparse_tile_rows) {
//                GNOpTile<SM, DM> tile_info;
//
//                tile_info.srows_start = i;
//                tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j)->nrows());
//
//                gSpMM_row_tiled_split(adj_vec.at(j),
//                                      inp_dense,
//                                      out_dense,
//                                      wsum_aggr,
//                                      k,
//                                      dense_tile_cols,
//                                      &tile_info);
//            }
//        }
//    }
//}

template<class SM, class DM>
void tile_ii_jj_i_j_kv(std::vector<SM *> adj_vec,
                       DM *inp_dense,
                       DM *out_dense,
                       typename SM::itype sparse_tile_rows) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
    // TODO assume that all the column tile segments have the same number of rows
    auto nrows = adj_vec.at(0)->nrows();

#ifdef LO_II_P
    //#pragma omp parallel for schedule(guided)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (iT i = 0; i < nrows; i += sparse_tile_rows) {
        GNOpTile<SM, DM> tile_info;

        tile_info.srows_start = i;
        tile_info.srows_end = std::min(i + sparse_tile_rows, nrows);

        for (diT j = 0; j < adj_vec.size(); j += 1) {
            gSpMM_row_tiled(adj_vec.at(j),
                            inp_dense,
                            out_dense,
                            wsum_aggr,
                            &tile_info);
        }
    }
}

template<class SM, class DM>
void tile_seg_iip_jj_i_j_kv(std::vector<SM *> adj_vec,
                            DM *inp_dense,
                            DM *out_dense,
                            typename SM::itype sparse_tile_rows,
                            typename SM::itype segment_size) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    iT nrows = adj_vec.at(0)->nrows();

    auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
    for (iT seg = 0; seg < adj_vec.size(); seg += segment_size) {
#pragma omp parallel for schedule(dynamic, 1)
        for (iT i = 0; i < nrows; i += sparse_tile_rows) {
            for (iT j = seg; j < std::min((iT) adj_vec.size(), seg + segment_size); j += 1) {
                GNOpTile<SM, DM> tile_info;

                tile_info.srows_start = i;
                tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j)->nrows());

                gSpMM_row_tiled(adj_vec.at(j),
                                inp_dense,
                                out_dense,
                                wsum_aggr,
                                &tile_info);
            }
        }
    }
}

template<class SM, class DM>
void tile_kk_jj_iip_i_j_kv(std::vector<SM *> adj_vec,
                           DM *inp_dense,
                           DM *out_dense,
                           typename SM::itype sparse_tile_rows,
                           typename DM::itype dense_tile_cols) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
    for (diT k = 0; k < inp_dense->ncols(); k += dense_tile_cols) {
        diT dcols_len = std::min(dense_tile_cols, inp_dense->ncols() - k);
        for (diT j = 0; j < adj_vec.size(); j += 1) {
#pragma omp parallel for schedule(dynamic, 1)
            for (iT i = 0; i < adj_vec.at(j)->nrows(); i += sparse_tile_rows) {
                GNOpTile<SM, DM> tile_info;

                tile_info.srows_start = i;
                tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j)->nrows());

                gSpMM_slice_row_tiled(adj_vec.at(j),
                                      inp_dense,
                                      out_dense,
                                      wsum_aggr,
                                      &tile_info,
                                      k,
                                      dcols_len);
            }
        }
    }
}

template<class SM, class DM>
void tile_jj_iip_i_j_kv_x_weight(std::vector<SM *> adj_vec,
                                 DM *inp_dense,
                                 DM *weight,
                                 DM *bias,
                                 DM *inter_dense,
                                 DM *out_dense,
                                 typename SM::itype sparse_tile_rows) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dvT *bias_vals_ptr = bias->vals_ptr();
    dvT *out_vals_ptr = out_dense->vals_ptr();
    diT out_ncols = out_dense->ncols();
    diT out_nrows = out_dense->nrows();

#pragma omp parallel for
    for (diT i_i = 0; i_i < out_nrows; i_i++) {
        dnT out_row_offset = i_i * out_ncols;
        for (diT i_j = 0; i_j < out_ncols; i_j++) {
            out_vals_ptr[out_row_offset + i_j] = bias_vals_ptr[i_j];
        }
    }

    auto wsum_aggr = wsumAgg<dvT, vT, dnT>;
    for (diT j = 0; j < adj_vec.size(); j += 1) {
#pragma omp parallel for schedule(dynamic, 1)
        for (iT i = 0; i < adj_vec.at(j)->nrows(); i += sparse_tile_rows) {
            GNOpTile<SM, DM> tile_info;

            tile_info.srows_start = i;
            tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j)->nrows());

#ifdef AUT_1
            gSpMM_row_tiled_x_weight(adj_vec.at(j),
                                     inp_dense,
                                     weight,
                                     bias,
                                     inter_dense,
                                     out_dense,
                                     wsum_aggr,
                                     &tile_info);
#elif AUT_2
            gSpMM_row_tiled_t_weight(adj_vec.at(j),
                                     inp_dense,
                                     weight,
                                     bias,
                                     inter_dense,
                                     out_dense,
                                     wsum_aggr,
                                     &tile_info);
#endif
        }
    }
}

template<class SM, class DM>
void tile_jj_iip_i_j_kv_p_weight(std::vector<SM *> adj_vec,
                                 DM *inp_dense,
                                 DM *weight,
                                 DM *bias,
                                 DM *inter_dense,
                                 DM *out_dense,
                                 typename SM::itype sparse_tile_rows) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    dvT *bias_vals_ptr = bias->vals_ptr();
    dvT *out_vals_ptr = out_dense->vals_ptr();
    diT out_ncols = out_dense->ncols();
    diT out_nrows = out_dense->nrows();

#pragma omp parallel for schedule(static)
    for (diT i_i = 0; i_i < out_nrows; i_i++) {
        dnT out_row_offset = i_i * out_ncols;
        for (diT i_k = 0; i_k < out_ncols; i_k++) {
            out_vals_ptr[out_row_offset + i_k] = bias_vals_ptr[i_k];
        }
    }

#ifdef TASK_SEP
#pragma omp parallel
    {
#pragma omp master
        {
            auto wsum_aggr = wsumAgg<dvT, vT, dnT>;

#ifdef REV_JJ
            for (diT j = adj_vec.size(); j > 0; j -= 1) {
#else
                for (diT j = 1; j < adj_vec.size() + 1; j += 1) {
#endif

//                std::cout << "Comes here1" << std::endl;
                for (iT i = 0; i < adj_vec.at(j - 1)->nrows(); i += sparse_tile_rows) {
                    GNOpTile<SM, DM> tile_info;

                    tile_info.srows_start = i;
                    tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j - 1)->nrows());

                    auto adj = adj_vec.at(j - 1);

#pragma omp task
                    {
                        if (adj->offset_ptr()[tile_info.srows_start] != adj->offset_ptr()[tile_info.srows_end + 1]) {
                            gSpMM_row_tiled_cs_weight(adj_vec.at(j - 1),
                                                      inp_dense,
                                                      inter_dense,
                                                      wsum_aggr,
                                                      &tile_info);
                        }
                    }
                }

//                std::cout << "Comes here2" << std::endl;
//#pragma omp taskwait

                for (iT i = 0; i < adj_vec.at(j - 1)->nrows(); i += sparse_tile_rows) {
                    GNOpTile<SM, DM> tile_info;

                    tile_info.srows_start = i;
                    tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j - 1)->nrows());
#pragma omp task
                    {
                        gSpMM_row_tiled_cw_weight(adj_vec.at(j - 1),
                                                  weight,
                                                  inter_dense,
                                                  out_dense,
                                                  &tile_info);
                    }
                }

                if (j == 0) {
                    break;
                }
            }
//            std::cout << "Comes here3.5" << std::endl;
        }
//        std::cout << "Comes here3.6" << std::endl;
    }
//    std::cout << "Comes here4" << std::endl;
#else
//#pragma omp parallel
//    {
//#pragma omp master
//        {
    auto wsum_aggr = wsumAgg<dvT, vT, dnT>;

    // TODO if you are going to reverse jj, then you need to add the -1 to the start not the end
#ifdef REV_JJ
    for (diT j = adj_vec.size(); j > 0; j -= 1) {
#else
    for (diT j = 1; j < adj_vec.size() + 1; j += 1) {
#endif
//#pragma omp taskloop
//#pragma omp parallel for schedule(guided)
#pragma omp parallel for schedule(dynamic, 1)
        for (iT i = 0; i < adj_vec.at(j - 1)->nrows(); i += sparse_tile_rows) {
            GNOpTile<SM, DM> tile_info;

            tile_info.srows_start = i;
            tile_info.srows_end = std::min(i + sparse_tile_rows, adj_vec.at(j - 1)->nrows());

#ifdef NEXT_GEMM
#ifdef REV_JJ
            if (j == 1){
                gSpMM_row_tiled_p_weight(adj_vec.at(j - 1),
                                     inp_dense,
                                     weight,
                                     bias,
                                     inter_dense,
                                     out_dense,
                                     wsum_aggr,
                                     &tile_info);
            } else {
                gSpMM_row_tiled_p_weight_cts(adj_vec.at(j - 1),
                                     inp_dense,
                                     weight,
                                     bias,
                                     inter_dense,
                                     out_dense,
                                     wsum_aggr,
                                     &tile_info);
            }
#else
            if (j == adj_vec.size()) {
                gSpMM_row_tiled_p_weight(adj_vec.at(j - 1),
                                         inp_dense,
                                         weight,
                                         bias,
                                         inter_dense,
                                         out_dense,
                                         wsum_aggr,
                                         &tile_info);
            } else {
                gSpMM_row_tiled_p_weight(adj_vec.at(j - 1),
                                             inp_dense,
                                             weight,
                                             bias,
                                             inter_dense,
                                             out_dense,
                                             wsum_aggr,
                                             &tile_info);
            }
#endif

#else
            gSpMM_row_tiled_p_weight(adj_vec.at(j - 1),
                                     inp_dense,
                                     weight,
                                     bias,
                                     inter_dense,
                                     out_dense,
                                     wsum_aggr,
                                     &tile_info);
#endif
        }
        if (j == 0) {
            break;
        }
    }
//        }
//    }
#endif
}


#endif // If for file