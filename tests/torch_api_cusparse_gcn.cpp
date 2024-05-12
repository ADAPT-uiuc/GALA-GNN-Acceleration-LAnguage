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

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUSPARSE_CHECK(func)                                                   \
  do {                                                                         \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


int main(int argc, char **argv) {
    // Input should be
    // 1. Input path
    // 2. Embedding size
    // 3. Colum tiling
    // 4. Row tiling
    // 5. Dense input slice
    // 6. Barriered execution
    // 7. Work division across multiple rows
    // 8. Reordering
    // 9. Prefetching

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    std::string path = argv[1];
    iT emb_size = stoi(string(argv[2]));

//    // Opt configurationsk
//    iT cols_per_tile = stoi(string(argv[3]));
//    iT rows_per_tile = stoi(string(argv[4]));
//    iT slice_size = stoi(string(argv[5]));
//    bool reord_mtx = stoi(string(argv[8]));
////    bool prefetch_data = stoi(string(argv[9])); // TODO

    // Timing configs
    int num_iters = stoi(string(argv[3]));

    // Const settings
    int skip_cache_warmup = 5;

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
    readSM<SM>(filename, &adj);
#endif
    adj.set_all(1);

    // Adj info
    iT nrows = adj.nrows();
    iT ncols = adj.ncols();
    nT nvals = adj.nvals();

//    if (reord_mtx) {
//        std::unique_ptr<vint[]> perm_rabbit;
//        auto nvals_var = adj.nvals();
//        auto nrows_var = adj.nrows();
//        iT *col_ids_var = adj.ids_ptr();
//        auto vals_var = adj.vals_ptr();
//        iT *row_ids_var;
//        get_row_ids<SM>(&adj, row_ids_var);
//        get_perm_graph<SM>(nrows_var, nvals_var, row_ids_var, col_ids_var, vals_var, perm_rabbit);
//        iT *perm = (iT *) aligned_alloc(64, sizeof(iT) * nrows);
//        for (iT p_i = 0; p_i < nrows; p_i++) {
//            perm[p_i] = (iT) perm_rabbit[p_i];
//        }
//        rowReorderToAdj(&adj, perm);
//    }

    std::vector<SM *> tiled_adj;
    tiled_adj.push_back(&adj);

//#ifdef ST_0
//    std::vector<iT> tile_offsets = static_ord_col_breakpoints<SM>(&adj, cols_per_tile);
//    ord_col_tiling(tile_offsets, tiled_adj, 0);
//#endif

    // Init input with random numbers
    DM input_emb;
    input_emb.build(adj.ncols(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
    for (diT i = 0; i < adj.nrows(); i++) {
        for (dnT j = 0; j < emb_size; j++) {
            input_emb.vals_ptr()[i * emb_size + j] = (dvT) (rand() % 100) / 100;
        }
    }

    DM out_emb;
    out_emb.build(adj.nrows(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);

    DM out_emb2;
    out_emb2.build(adj.nrows(), emb_size, DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);

    auto wsum_aggr = wsumAgg<val_t, val_t, ind2_t>;

    std::cout << adj.nrows() <<  " " << adj.ncols() <<  " "  << adj.nvals() << std::endl;

    float alpha = 1.0f;
    float beta = 1.0f;

    // Create the sparse / dense objects
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, nrows, nrows, nvals,
                                     offset_ptr, col_ptr, val_ptr,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, // Need to change these
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // Create dense matrix B
    CUSPARSE_CHECK(cusparseCreateDnMat(&matB, nrows, dcols, dcols, iden_ptr,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW)); // changed
    // Create dense matrix C
    CUSPARSE_CHECK(cusparseCreateDnMat(&matC, nrows, dcols, dcols, oden_array,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW)); // changed

    // allocate an external buffer if needed
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_CSR_ALG2,
            &bufferSize));
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    CUSPARSE_CHECK(cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_CSR_ALG2,
                                dBuffer));

    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC));

    int i;
    double start, end;
    std::vector<double> times_arr;
    for (i = 0; i < num_iters + skip_cache_warmup; i++) {
        out_emb.set_all(0);
        start = get_time();
//        tile_jj_ii_i_j_kv(tiled_adj,
//                          &input_emb,
//                          &out_emb,
//                          rows_per_tile,
//                          wsum_aggr);
        end = get_time();

        if (i >= skip_cache_warmup) {
            times_arr.push_back(end - start);
        }
    }

//    std::cout << adj.offset_ptr()[1] << " " << adj.offset_ptr()[2] << " " << adj.offset_ptr()[3] << " "
//              << adj.offset_ptr()[4] << " " << adj.offset_ptr()[5] << std::endl;
//    std::cout << adj.ids_ptr()[1] << " " << adj.ids_ptr()[2] << " " << adj.ids_ptr()[3] << " "
//              << adj.ids_ptr()[4] << " " << adj.ids_ptr()[5] << std::endl;
//    std::cout << out_emb2.vals_ptr()[0] << " " << out_emb2.vals_ptr()[0 + input_emb.ncols() * 8] << std::endl;
//    std::cout << out_emb.vals_ptr()[0] << " " << out_emb.vals_ptr()[0 + out_emb.ncols() * 8] << std::endl;

//    for (nT j = 0; j < nvals; j++) {
//        if (out_emb.vals_ptr()[j] != out_emb2.vals_ptr()[j]) {
//            std::cout << "The results don't match at: " << j << ", " << out_emb.vals_ptr()[j] << ", "
//                      << out_emb2.vals_ptr()[j] << std::endl;
//            break;
//        }
//    }
    std::cout << calc_mean(times_arr) << "," << calc_std(times_arr) << std::endl;
    return 0;
}