
#include <omp.h>

//Customizable vector aggregators for gSpMM.
//The SpMM aggregators can be redefined at a set and not pair level.
//Each aggregator is subject to individual optimizations but feature-level parallelism 
//is a common notion (although LSTM aggregator perplexes the situation).

#ifndef AGGREGATORS_H
#define AGGREGATORS_H

template<class dvT, class vT, class dnT>
inline void wsumAgg(dvT *accum, dvT *to_add, vT weight, dnT length) {
    //Weighted sum aggregator.
    // TODO try moving the aggregator, to a variable and then writing back once it's finished
#ifdef LO_K_P
#pragma omp parallel for schedule(static)
    for (dnT jj = 0; jj < length; jj+=16) {
        dnT mx_j = std::min(jj+16, length);
#pragma omp simd
        for (dnT j = jj; j < mx_j; j++) {
            accum[j] += weight * to_add[j];
        }
    }
#else
//#pragma omp simd
    for (dnT j = 0; j < length; j++) {
        accum[j] += weight * to_add[j];
    }
#endif
}


template<class dvT, class vT, class dnT>
inline void maxAgg(dvT *accum, dvT *to_add, vT weight, dnT length) {
    //Max aggregator.

#pragma omp simd
    for (dnT j = 0; j < length; j++) {
        accum[j] = accum[j] >= to_add[j] ? accum[j] : to_add[j];
    }

}

template<class dvT, class dnT>
inline void sumAgg(dvT *accum, dvT *to_add, dnT length) {
    //Sum aggregator.

#pragma omp simd
    for (dnT j = 0; j < length; j++) {
        accum[j] += to_add[j];
    }
}

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

#endif