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

#ifdef MKL
#include <mkl.h>
#endif
#ifdef OPB
#include <cblas.h>
#endif

#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

// static int count50 = 0;
// static int count75 = 0;
// static int count90 = 0;

template<class DM>
void MMb(const DM *A, const DM *B, DM *out, const DM *bias) {
    //This is not GEMM since the operation is out=A*B
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    diT A_nrows = A->nrows();
    diT B_ncols = B->ncols();
    diT A_ncols = A->ncols();
    dvT *A_vals_ptr = A->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    dvT *bias_vals_ptr = bias->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    diT out_ncols = out->ncols();

    if (B->type() == DM::DENSE_MTX_TYPE::RM) {
#pragma omp parallel for
        for (diT i_i = 0; i_i < A_nrows; i_i++) {
            dnT A_row_offset = i_i * A_ncols;
            dnT out_row_offset = i_i * out_ncols;
            for (diT i_j = 0; i_j < B_ncols; i_j++) {
                dvT sum = bias_vals_ptr[i_j];;
                // TODO Manually create a vectorization here?
                for (diT i_k = 0; i_k < A_ncols; i_k++) {
                    dvT v_i = A_vals_ptr[(dnT) (A_row_offset + i_k)];
                    dvT v_j = B_vals_ptr[(dnT) (i_k * B_ncols + i_j)];
                    sum += v_i * v_j;
                }
                out_vals_ptr[out_row_offset + i_j] = sum;
            }
        }
    } else if (B->type() == DM::DENSE_MTX_TYPE::CM) {
#pragma omp parallel for
        for (diT i_i = 0; i_i < A_nrows; i_i++) {
            dnT A_row_offset = i_i * A_ncols;
            dnT out_row_offset = i_i * out_ncols;
            for (diT i_j = 0; i_j < B_ncols; i_j++) {
                dvT sum = bias_vals_ptr[i_j];
                dnT B_col_offset = i_j * A_ncols;
                for (diT i_k = 0; i_k < A_ncols; i_k++) {
                    dvT v_i = A_vals_ptr[(dnT) (A_row_offset + i_k)];
                    dvT v_j = B_vals_ptr[(dnT) (i_k + B_col_offset)];
                    sum += v_i * v_j;
                }
                out_vals_ptr[out_row_offset + i_j] = sum;
            }
        }
    }
}

template<class DM>
void MM(const DM *A, const DM *B, DM *out) {
    //This is not GEMM since the operation is out=A*B
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    diT A_nrows = A->nrows();
    diT B_nrows = B->nrows();
    diT A_ncols = A->ncols();
    diT B_ncols = B->ncols();
    dvT *A_vals_ptr = A->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    diT out_ncols = out->ncols();

    if (B->type() == DM::DENSE_MTX_TYPE::RM) {
#pragma omp parallel for
        for (diT i_i = 0; i_i < A_nrows; i_i++) {
            dnT A_row_offset = i_i * A_ncols;
            dnT out_row_offset = i_i * out_ncols;
            for (diT i_j = 0; i_j < B_ncols; i_j++) {
                dvT sum = 0;
                // TODO Manually create a vectorization here?
                for (diT i_k = 0; i_k < A_ncols; i_k++) {
                    dvT v_i = A_vals_ptr[(dnT) (A_row_offset + i_k)];
                    dvT v_j = B_vals_ptr[(dnT) (i_k * B_ncols + i_j)];
                    sum += v_i * v_j;
                }
                out_vals_ptr[out_row_offset + i_j] = sum;
            }
        }
    } else if (B->type() == DM::DENSE_MTX_TYPE::CM) {
#pragma omp parallel for
        for (diT i_i = 0; i_i < A_nrows; i_i++) {
            dnT A_row_offset = i_i * A_ncols;
            dnT out_row_offset = i_i * out_ncols;
            for (diT i_j = 0; i_j < B_ncols; i_j++) {
                dvT sum = 0;
                dnT B_col_offset = i_j * B_nrows;
                for (diT i_k = 0; i_k < A_ncols; i_k++) {
                    dvT v_i = A_vals_ptr[(dnT) (A_row_offset + i_k)];
                    dvT v_j = B_vals_ptr[(dnT) (i_k + B_col_offset)];
                    sum += v_i * v_j;
                }
                out_vals_ptr[out_row_offset + i_j] = sum;
            }
        }
    }
}

template<class DM>
void MM_additive(const DM *A, const DM *B, DM *out) {
    //This is not GEMM since the operation is out=A*B
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    diT A_nrows = A->nrows();
    diT B_nrows = B->nrows();
    diT A_ncols = A->ncols();
    diT B_ncols = B->ncols();
    dvT *A_vals_ptr = A->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    diT out_ncols = out->ncols();

    if (B->type() == DM::DENSE_MTX_TYPE::RM) {
#pragma omp parallel for
        for (diT i_i = 0; i_i < A_nrows; i_i++) {
            dnT A_row_offset = i_i * A_ncols;
            dnT out_row_offset = i_i * out_ncols;
            for (diT i_j = 0; i_j < B_ncols; i_j++) {
                dvT sum = out_vals_ptr[out_row_offset + i_j];
                // TODO Manually create a vectorization here?
                for (diT i_k = 0; i_k < A_ncols; i_k++) {
                    dvT v_i = A_vals_ptr[(dnT) (A_row_offset + i_k)];
                    dvT v_j = B_vals_ptr[(dnT) (i_k * B_ncols + i_j)];
                    sum += v_i * v_j;
                }
                out_vals_ptr[out_row_offset + i_j] = sum;
            }
        }
    } else if (B->type() == DM::DENSE_MTX_TYPE::CM) {
#pragma omp parallel for
        for (diT i_i = 0; i_i < A_nrows; i_i++) {
            dnT A_row_offset = i_i * A_ncols;
            dnT out_row_offset = i_i * out_ncols;
            for (diT i_j = 0; i_j < B_ncols; i_j++) {
                dvT sum = out_vals_ptr[out_row_offset + i_j];
                dnT B_col_offset = i_j * B_nrows;
                for (diT i_k = 0; i_k < A_ncols; i_k++) {
                    dvT v_i = A_vals_ptr[(dnT) (A_row_offset + i_k)];
                    dvT v_j = B_vals_ptr[(dnT) (i_k + B_col_offset)];
                    sum += v_i * v_j;
                }
                out_vals_ptr[out_row_offset + i_j] = sum;
            }
        }
    }
}


template<class DM>
void MMbroacast_row(const DM *A, const DM *B, DM *out) {
    //Do MM broadcast operations
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    diT A_nrows = A->nrows();
    diT A_ncols = A->ncols();
    dvT *A_vals_ptr = A->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    diT out_ncols = out->ncols();

#pragma omp parallel for schedule(static)
    for (diT i_i = 0; i_i < A_nrows; i_i++) {
        dnT A_row_offset = i_i * A_ncols;
        dnT out_row_offset = i_i * out_ncols;
        dvT v_j = B_vals_ptr[(dnT) (i_i)];
#pragma omp simd
        for (diT i_j = 0; i_j < A_ncols; i_j++) {
            out_vals_ptr[(dnT) (out_row_offset + i_j)] = A_vals_ptr[(dnT) (A_row_offset + i_j)] * v_j;
        }
    }
}

template<class DM>
void MMbroacast_col(const DM *A, const DM *B, DM *out) {
    //Do MM broadcast operations
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    diT A_nrows = A->nrows();
    diT A_ncols = A->ncols();
    dvT *A_vals_ptr = A->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    diT out_ncols = out->ncols();

#pragma omp parallel for
    for (diT i_i = 0; i_i < A_nrows; i_i++) {
        dnT A_row_offset = i_i * A_ncols;
        dnT out_row_offset = i_i * out_ncols;
#pragma omp simd
        for (diT i_j = 0; i_j < A_ncols; i_j++) {
            dvT v_i = A_vals_ptr[(dnT) (A_row_offset + i_j)];
            dvT v_j = B_vals_ptr[(dnT) (i_j)];
            out_vals_ptr[out_row_offset + i_j] = v_i * v_j;
        }
    }
}

template<class DM>
void MMb_mkl(const DM *A, const DM *B, DM *out, const DM *bias) {
    typedef typename DM::itype diT; // Node IDs
    typedef typename DM::ntype dnT; // Edge IDs
    typedef typename DM::vtype dvT; // Value of nodes

    //First we load the output with the corresponding bias and then we perform Out=A*B+Out.
    diT A_nrows = A->nrows();
    diT B_ncols = B->ncols();
    dvT *bias_vals_ptr = bias->vals_ptr();
    diT A_ncols = A->ncols();
    dvT *A_vals_ptr = A->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    diT out_ncols = out->ncols();
    diT out_nrows = out->nrows();
    diT B_nrows = B->nrows();

#pragma omp parallel for
    for (diT i_i = 0; i_i < out_nrows; i_i++) {
        dnT out_row_offset = i_i * out_ncols;
        for (diT i_j = 0; i_j < out_ncols; i_j++) {
            out_vals_ptr[out_row_offset + i_j] = bias_vals_ptr[i_j];
        }
    }
    // auto test1 = (float*)aligned_alloc(64, sizeof(float)*25);
    // auto test2 = (float*)aligned_alloc(64, sizeof(float)*25);
    // auto test3 = (float*)aligned_alloc(64, sizeof(float)*25);

    // auto test1 = (double*)aligned_alloc(64, sizeof(double)*25);
    // auto test2 = (double*)aligned_alloc(64, sizeof(double)*25);
    // auto test3 = (double*)aligned_alloc(64, sizeof(double)*25);
    // double one_v = 1;

    if (B->type() == DM::DENSE_MTX_TYPE::RM) {
#ifdef GEMM_MKL
        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 5, 5, 5, one_v, test1, 5,
        //     test2, 5, one_v, test3, 5);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                    B_vals_ptr, B_ncols, 1.0f, out_vals_ptr, B_ncols);
#elif GEMM_OPB
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                    B_vals_ptr, B_ncols, 1.0f, out_vals_ptr, B_ncols);
#endif
    } else if (B->type() == DM::DENSE_MTX_TYPE::CM) {
#ifdef GEMM_MKL
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                    B_vals_ptr, B_nrows, 1.0f, out_vals_ptr, B_ncols);
#elif GEMM_OPB
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                    B_vals_ptr, B_nrows, 1.0f, out_vals_ptr, B_ncols);
#endif
    }
}

template<class DM>
void MM_mkl(const DM *A, const DM *B, DM *out) {
    typedef typename DM::itype diT; // Node IDs
    typedef typename DM::ntype dnT; // Edge IDs
    typedef typename DM::vtype dvT; // Value of nodes

    diT A_nrows = A->nrows();
    diT B_ncols = B->ncols();
    diT A_ncols = A->ncols();
    dvT *A_vals_ptr = A->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    diT out_ncols = out->ncols();
    diT out_nrows = out->nrows();
    diT B_nrows = B->nrows();

    // auto test1 = (float*)aligned_alloc(64, sizeof(float)*25);
    // auto test2 = (float*)aligned_alloc(64, sizeof(float)*25);
    // auto test3 = (float*)aligned_alloc(64, sizeof(float)*25);

    // auto test1 = (double*)aligned_alloc(64, sizeof(double)*25);
    // auto test2 = (double*)aligned_alloc(64, sizeof(double)*25);
    // auto test3 = (double*)aligned_alloc(64, sizeof(double)*25);
    // double one_v = 1;

    if (B->type() == DM::DENSE_MTX_TYPE::RM) {
#ifdef GEMM_MKL
        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 5, 5, 5, one_v, test1, 5,
        //     test2, 5, one_v, test3, 5);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                    B_vals_ptr, B_ncols, 0.0f, out_vals_ptr, B_ncols);
#elif GEMM_OPB
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                    B_vals_ptr, B_ncols, 0.0f, out_vals_ptr, B_ncols);
#endif
    } else if (B->type() == DM::DENSE_MTX_TYPE::CM) {
        // TODO this is buggy
#ifdef GEMM_MKL
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                    B_vals_ptr, B_nrows, 0.0f, out_vals_ptr, B_ncols);
#elif GEMM_OPB
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                    B_vals_ptr, B_nrows, 0.0f, out_vals_ptr, B_ncols);
#endif
    }

}

template<class DM>
void MM_mkl_additive(const DM *A, const DM *B, DM *out) {
    typedef typename DM::itype diT; // Node IDs
    typedef typename DM::ntype dnT; // Edge IDs
    typedef typename DM::vtype dvT; // Value of nodes

    diT A_nrows = A->nrows();
    diT B_ncols = B->ncols();
    diT A_ncols = A->ncols();
    dvT *A_vals_ptr = A->vals_ptr();
    dvT *B_vals_ptr = B->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();
    diT out_ncols = out->ncols();
    diT out_nrows = out->nrows();
    diT B_nrows = B->nrows();

    if (B->type() == DM::DENSE_MTX_TYPE::RM) {
#ifdef GEMM_MKL
        if (sizeof(dvT) == 8) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                        B_vals_ptr, B_ncols, 1.0f, out_vals_ptr, B_ncols);
        } else {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                        B_vals_ptr, B_ncols, 1.0f, out_vals_ptr, B_ncols);
        }

#elif GEMM_OPB
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                    B_vals_ptr, B_ncols, 1.0f, out_vals_ptr, B_ncols);
#endif
    } else if (B->type() == DM::DENSE_MTX_TYPE::CM) {
        // TODO this is buggy
#ifdef GEMM_MKL
        if (sizeof(dvT) == 8) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                        B_vals_ptr, B_nrows, 1.0f, out_vals_ptr, B_ncols);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                        B_vals_ptr, B_nrows, 1.0f, out_vals_ptr, B_ncols);
        }
#elif GEMM_OPB
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_nrows, B_ncols, A_ncols, 1.0f, A_vals_ptr, A_ncols,
                    B_vals_ptr, B_nrows, 1.0f, out_vals_ptr, B_ncols);
#endif
    }

}

template<class DM, class Function>
void UEwD(DM *in, DM *out, Function uoperator) {
    //Unary element wise operation for Dense Matrices.

    typedef typename DM::itype diT; // Node IDs
    typedef typename DM::ntype dnT; // Edge IDs
    typedef typename DM::vtype dvT; // Value of nodes

    diT in_nrows = in->nrows();
    diT in_ncols = in->ncols();
    dvT *in_vals_ptr = in->vals_ptr();
    dvT *out_vals_ptr = out->vals_ptr();


#pragma omp parallel for
    for (diT i_i = 0; i_i < in_nrows; i_i++) {
        dnT row_offset = i_i * in_ncols;
        for (diT i_j = 0; i_j < in_ncols; i_j++) {
            out_vals_ptr[row_offset + i_j] = uoperator(in_vals_ptr[row_offset + i_j]);
        }
    }
}

#endif // If for file