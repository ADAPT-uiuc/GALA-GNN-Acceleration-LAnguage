#ifndef _CSR_MATRIX_H
#define _CSR_MATRIX_H

#include "../utils/info_error.h"
#include "../utils/mtx_sort.h"
#include "../utils/threading_utils.h"
//#include <immintrin.h>
// #include "matrix.h"
#include "matrix_prop.h"
// #include "../operations/"
// struct MatrixProperties;
// template <typename I_, typename V_, template <class A> class Alloc>
// class Vector;

#include <string>

using std::string;

#ifdef SparseLibDBG
#define CSR_MTX_DBG
#endif

#ifdef CSR_MTX_DBG
#define CSR_MTX_DBG_MSG std::cout << "CSR_MTX_" << __LINE__ << ": "
#define CSR_MTX_DBG_MSG2 std::cout
#else
#define CSR_MTX_DBG_MSG \
  if (0) std::cout << "CSR_MTX_" << __LINE__ << ": "
#define CSR_MTX_DBG_MSG2 \
  if (0) std::cout
#endif

template<typename I, typename N>
bool compSortingPair2(const std::pair<I, N> &a, const std::pair<I, N> &b) {
    return a.second > b.second;
}

template<typename I, typename N>
bool compSortingPair3(const std::pair<I, N> &a, const std::pair<I, N> &b) {
    return a.second < b.second;
}

template<typename I_, typename N_, typename V_, template<class A> class Alloc = std::allocator>
class CSRCMatrix {
public:
    typedef V_ vtype;
    typedef I_ itype;
    typedef N_ ntype;

    CSRCMatrix() : nrows_(0), ncols_(0), nvals_(0), comp_ids_(nullptr), ids_(nullptr), vals_(nullptr),
                   offset_(nullptr) {}

    ~CSRCMatrix() { clear(); }

    // doesn't copy the arrays
    INFO import_mtx(I_ nrows, I_ ncols, N_ nvals, I_ *ids, V_ *vals, N_ *offset, CSRC_TYPE type,
                    MatrixProperties mtx_prop = MatrixProperties()) {
        nrows_ = nrows;
        ncols_ = ncols;
        nvals_ = nvals;
        ids_ = ids;
        vals_ = vals;
        offset_ = offset;
        mtx_prop_ = mtx_prop;
        type_ = type;
        return INFO::SUCCESS;
    }

    INFO import_mtx(I_ nrows, I_ ncols, N_ nvals, I_ *ids, V_ *vals, N_ *offset, I_ *rows, CSRC_TYPE type,
                    MatrixProperties mtx_prop = MatrixProperties()) {
        nrows_ = nrows;
        ncols_ = ncols;
        nvals_ = nvals;
        ids_ = ids;
        vals_ = vals;
        offset_ = offset;
        row_ids_ = rows;
        mtx_prop_ = mtx_prop;
        type_ = type;
        return INFO::SUCCESS;
    }

    INFO import_work_mask(bool *work) {
        do_GEMM_work_ = work;
        return INFO::SUCCESS;
    }

    INFO clone_mtx(I_ nrows, I_ ncols, N_ nvals, I_ *ids, V_ *vals, N_ *offset, CSRC_TYPE type,
                   MatrixProperties mtx_prop = MatrixProperties()) {
        nrows_ = nrows;
        ncols_ = ncols;
        nvals_ = nvals;

        I_ *new_ids_ = (I_ *) aligned_alloc(64, (nvals_) * sizeof(I_));
        V_ *new_vals_ = (V_ *) aligned_alloc(64, (nvals_) * sizeof(V_));
        N_ *new_offset_ = (N_ *) aligned_alloc(64, (nrows_ + 1) * sizeof(N_));

        std::copy(ids, ids + nvals_, new_ids_);
        std::copy(vals, vals + nvals_, new_vals_);
        std::copy(offset, offset + nrows_ + 1, new_offset_);

        ids_ = new_ids_;
        vals_ = new_vals_;
        offset_ = new_offset_;

        mtx_prop_ = mtx_prop;
        type_ = type;
        return INFO::SUCCESS;
    }

    INFO export_mtx(I_ &nrows, I_ &ncols, N_ &nvals, I_ *&ids, V_ *&vals, N_ *&offset, CSRC_TYPE &type,
                    MatrixProperties &mtx_prop) {
        nrows = nrows_;
        ncols = ncols_;
        nvals = nvals_;
        ids = ids_;
        vals = vals_;
        offset = offset_;
        mtx_prop = mtx_prop_;
        type = type_;
        return INFO::SUCCESS;
    }

    INFO export_csr(I_ &nrows, I_ &ncols, N_ &nvals, I_ *&ids, V_ *&vals, N_ *&offset) {
        CSRC_TYPE type;
        MatrixProperties prop;
        return export_mtx(nrows, ncols, nvals, ids, vals, offset, type, prop);
    }

    INFO export_csc(I_ &nrows, I_ &ncols, N_ &nvals, I_ *&ids, V_ *&vals, N_ *&offset) {
        CSRC_TYPE type;
        MatrixProperties prop;
        return export_mtx(nrows, ncols, nvals, ids, vals, offset, type, prop);
    }

    INFO import_csr(I_ nrows, I_ ncols, N_ nvals, I_ *ids, V_ *vals, N_ *offset) {
        return import_mtx(nrows, ncols, nvals, ids, vals, offset, CSRC_TYPE::CSR);
    }

    INFO import_csc(I_ nrows, I_ ncols, N_ nvals, I_ *ids, V_ *vals, N_ *offset) {
        return import_mtx(nrows, ncols, nvals, ids, vals, offset, CSRC_TYPE::CSC);
    }

    INFO import_dcsr(I_ nrows, I_ ncols, N_ nvals, I_ *ids, V_ *vals, N_ *offset, I_ *rows) {
        return import_mtx(nrows, ncols, nvals, ids, vals, offset, rows, CSRC_TYPE::DCSR);
    }

    INFO
    build(I_ nrows, I_ ncols, N_ nvals, I_ *&row_ids, I_ *&col_ids, V_ *&vals, CSRC_TYPE type, int8_t is_sorted = -1,
          bool has_dup = true, MatrixProperties mtx_prop = MatrixProperties()) {
        CSR_MTX_DBG_MSG << "CSR build nr: " << nrows << " nc: " << ncols << " nnz: " << nvals << std::endl;
//        double s, e;
//        s = get_time();
        type_ = type;
        nrows_ = nrows;
        ncols_ = ncols;
        nvals_ = nvals;
        mtx_prop_ = mtx_prop;
        type_ = type;

        N_ *counts;  // counts to calculate offsets

        I_ nids;        // either nrows or ncols
        I_ *ids_local;  // row_ids or col_ids
        I_ *ids2 = nullptr;
        // allocate counts, offset_, ids_, vals_
        try {
            if (type == CSRC_TYPE::CSR || type == CSRC_TYPE::HCSR) {
                offset_ = offset_alloc_.allocate(nrows_ + 1);
                counts = offset_alloc_.allocate(nrows_);
            } else {
                offset_ = offset_alloc_.allocate(ncols_ + 1);
                counts = offset_alloc_.allocate(ncols_);
            }
            ids_ = ids_alloc_.allocate(nvals_);
#ifdef CSR_MTX_DBG
            ids2 = ids_alloc_.allocate(nvals_);
#endif
            if (vals != nullptr) vals_ = vals_alloc_.allocate(nvals_);
        } catch (const std::bad_array_new_length &e) {
#ifdef CSR_MTX_DBG
            if (ids2 != nullptr) {
              ids_alloc_.deallocate(ids2, nvals_);
              ids2 = nullptr;
            }
#endif
            clear();
            return INFO::OUT_OF_MEMORY;
        } catch (const std::bad_alloc &e) {
            // ids_alloc_.deallocate(ids2, nvals_);
#ifdef CSR_MTX_DBG
            if (ids2 != nullptr) {
              ids_alloc_.deallocate(ids2, nvals_);
              ids2 = nullptr;
            }
#endif
            clear();
            return INFO::OUT_OF_MEMORY;
        }

        if (type == CSRC_TYPE::CSR || type == CSRC_TYPE::HCSR) {
            count_atomic(row_ids, counts, nrows_, nvals_);
            partial_sum(counts, offset_, nrows_);
            // for (int i = 0; i < 20; i++)
            //   CSR_MTX_DBG_MSG << "Offset[" << i << "] " << offset_[i] << " - " << offset_[i + 1] << " "
            //                   << "Counts[" << i << "] " << counts[i] << std::endl;

            assert(offset_[nrows_] == nvals_);

            // copy the values from row_ids and vals to ids_ and vals_
            if (vals != nullptr) {
#ifndef CSR_MTX_DBG
                count_sort_place_2arr(row_ids, offset_, counts, nrows, nvals, col_ids, vals, ids_, vals_);
#else
                count_sort_place_3arr(row_ids, offset_, counts, nrows, nvals, col_ids, row_ids, vals, ids_, ids2, vals_);
#endif
            } else {
#ifndef CSR_MTX_DBG
                count_sort_place_1arr(row_ids, offset_, counts, nrows, nvals, col_ids, ids_);
#else
                count_sort_place_2arr(row_ids, offset_, counts, nrows, nvals, col_ids, row_ids, ids_, ids2);
#endif
            }
#ifdef CSR_MTX_DBG
#pragma omp parallel for
            for (I_ r = 0; r < nrows; r++) assert(offset_[r + 1] == counts[r]);
#endif
            CSR_MTX_DBG_MSG << "Count sort place done" << std::endl;
            // template <typename ID_, typename N_, typename K_, typename I1_>
            // void sort_range1arr(ID_* ids, N_* ranges, K_ nranges, N_ nvals, I1_* arr1, I1_* arr1_c)
            if (vals == nullptr) {
#ifndef CSR_MTX_DBG
                sort_range1arr(ids_, offset_, nrows_, nvals_, ids_, col_ids);

#else
                sort_range2arr(ids_, offset_, nrows_, nvals_, ids_, ids2, col_ids, row_ids);
#endif
// copy back to ids_
#pragma omp parallel for
                for (N_ n = 0; n < nvals_; n++) {
                    ids_[n] = col_ids[n];
#ifdef CSR_MTX_DBG
                    ids2[n] = row_ids[n];
#endif
                }
            } else {
#ifndef CSR_MTX_DBG
                sort_range2arr(ids_, offset_, nrows_, nvals_, ids_, vals_, col_ids, vals);
#else
                sort_range3arr(ids_, offset_, nrows_, nvals_, ids_, ids2, vals_, col_ids, row_ids, vals);
#endif
#pragma omp parallel for
                for (N_ n = 0; n < nvals_; n++) {
                    ids_[n] = col_ids[n];
#ifdef CSR_MTX_DBG
                    ids2[n] = row_ids[n];
#endif
                    vals_[n] = vals[n];
                }
            }

            // for (int i = 0; i < 3; i++) {
            //   CSR_MTX_DBG_MSG << "Offset[" << i << "] " << offset_[i] << " - " << offset_[i + 1] << " "
            //                   << "Counts[" << i << "] " << counts[i] << std::endl;
            //   // for(N_ o=offset_[i]; o<offset_[i+1]; o++){
            //   //   std::cout<<"\t"<<i<<","<<ids_[o]<<std::endl;
            //   // }
            // }ords

            // for (int i = 0; i < 20; i++) std::cout << i << " ids: " << ids_[i] << std::endl;

#ifdef CSR_MTX_DBG
#pragma omp parallel for
            for (N_ e = 1; e < nvals_; e++) {
              assert(row_ids[e - 1] <= row_ids[e]);
              assert(e < offset_[row_ids[e] + 1]);
            }
#endif

            offset_alloc_.deallocate(counts, nrows_);
        }
        if (type == CSRC_TYPE::CSC || type == CSRC_TYPE::HCSC) {
            count_atomic(col_ids, counts, ncols_, nvals_);
            partial_sum(counts, offset_, ncols_);
            // for (int i = 0; i < 20; i++)
            //   CSR_MTX_DBG_MSG << "Offset[" << i << "] " << offset_[i] << " - " << offset_[i + 1] << " "
            //                   << "Counts[" << i << "] " << counts[i] << std::endl;

            assert(offset_[ncols_] == nvals_);

            // copy the values from row_ids and vals to ids_ and vals_
            if (vals != nullptr) {
#ifndef CSR_MTX_DBG
                count_sort_place_2arr(col_ids, offset_, counts, ncols_, nvals, row_ids, vals, ids_, vals_);
#else
                count_sort_place_3arr(col_ids, offset_, counts, ncols_, nvals, row_ids, col_ids, vals, ids_, ids2, vals_);
#endif
            } else {
#ifndef CSR_MTX_DBG
                count_sort_place_1arr(col_ids, offset_, counts, ncols_, nvals, row_ids, ids_);
#else
                count_sort_place_2arr(col_ids, offset_, counts, ncols_, nvals, row_ids, col_ids, ids_, ids2);
#endif
            }

#ifdef CSR_MTX_DBG
#pragma omp parallel for
            for (I_ c = 0; c < ncols; c++) assert(offset_[c + 1] == counts[c]);
#endif

            CSR_MTX_DBG_MSG << "Count sort place done" << std::endl;
            // template <typename ID_, typename N_, typename K_, typename I1_>
            // void sort_range1arr(ID_* ids, N_* ranges, K_ nranges, N_ nvals, I1_* arr1, I1_* arr1_c)
            if (vals == nullptr) {
#ifndef CSR_MTX_DBG
                sort_range1arr(ids_, offset_, ncols_, nvals_, ids_, row_ids);
#else
                sort_range2arr(ids_, offset_, ncols_, nvals_, ids_, ids2, row_ids, col_ids);
#endif
// copy back to ids_
#pragma omp parallel for
                for (N_ n = 0; n < nvals_; n++) {
                    ids_[n] = row_ids[n];
#ifdef CSR_MTX_DBG
                    ids2[n] = col_ids[n];
#endif
                }
            } else {
#ifndef CSR_MTX_DBG
                sort_range2arr(ids_, offset_, ncols_, nvals_, ids_, vals_, row_ids, vals);
#else
                sort_range3arr(ids_, offset_, ncols_, nvals_, ids_, ids2, vals_, row_ids, col_ids, vals);
#endif
#pragma omp parallel for
                for (N_ n = 0; n < nvals_; n++) {
                    ids_[n] = row_ids[n];
#ifdef CSR_MTX_DBG
                    ids2[n] = col_ids[n];
#endif
                    vals_[n] = vals[n];
                }
            }

            //       for (int i = 0; i < 3; i++) {
            //         CSR_MTX_DBG_MSG << "Offset[" << i << "] " << offset_[i] << " - " << offset_[i + 1] << " "
            //                         << "Counts[" << i << "] " << counts[i] << std::endl;
            // // for(N_ o=offset_[i]; o<offset_[i+1]; o++){
            // //   std::cout<<"\t"<<i<<","<<ids_[o]<<std::endl;
            // // }

            //       }

#ifdef CSR_MTX_DBG
#pragma omp parallel for
            for (N_ e = 1; e < nvals_; e++) {
              assert(col_ids[e - 1] <= col_ids[e]);
              assert(e < offset_[col_ids[e] + 1]);
            }
#endif

            // for (int i = 0; i < 20; i++) std::cout << i << " ids: " << ids_[i] << std::endl;

            offset_alloc_.deallocate(counts, ncols);
        }
// #ifdef CSR_MTX_DBG
//     if (ids2 != nullptr) {
//       ids_alloc_.deallocate(ids2, nvals_);
//       ids2 = nullptr;
//     }
// #endif

        //e = get_time();
        //std::cout << "TimeCSRbuild: " << (e - s) << std::endl;
        return INFO::SUCCESS;
    }

    INFO clear() {
        if (ids_ != nullptr) {
            ids_alloc_.deallocate(ids_, nvals_);
            ids_ = nullptr;
        }
        if (vals_ != nullptr) {
            vals_alloc_.deallocate(vals_, nvals_);
            vals_ = nullptr;
        }
        if (offset_ != nullptr) {
            if (type_ == CSRC_TYPE::CSR || type_ == CSRC_TYPE::HCSR) {
                offset_alloc_.deallocate(offset_, nrows_ + 1);
            }
            if (type_ == CSRC_TYPE::CSC || type_ == CSRC_TYPE::HCSC) {
                offset_alloc_.deallocate(offset_, ncols_ + 1);
            }
            offset_ = nullptr;
        }
        return INFO::SUCCESS;
    }

    void set_all(V_ val) {
        if (vals_ == nullptr) {
            vals_ = vals_alloc_.allocate(nvals_);
        }
#pragma omp parallel for
        for (N_ v = 0; v < nvals_; v++) {
            vals_[v] = val;
        }
    }

    // get functions
    inline I_ nrows() const { return nrows_; }

    inline I_ ncols() const { return ncols_; }

    inline N_ nvals() const { return nvals_; }

    inline CSRC_TYPE type() const { return type_; }

    inline I_ *ids_ptr() const { return ids_; }

    inline V_ *vals_ptr() const { return vals_; }

    inline N_ *offset_ptr() const { return offset_; }

    inline void set_mtx_name(string mtx_name) { mtx_name_ = mtx_name; }

    inline string mtx_name() const { return mtx_name_; }

    inline bool *work_ptr() const { return do_GEMM_work_; }

    // For row and column IDS of COO based ones

    inline I_ *col_ids_ptr() const { return col_ids_; }

    inline I_ *row_ids_ptr() const { return row_ids_; }

private:
    string mtx_name_;
    // MtxStats mtx_stats_;
    I_ nrows_;
    I_ ncols_;
    N_ nvals_;
    I_ *ids_;  // used for CSR
    I_ *idsw_;
    I_ *comp_ids_;  // used for hyper sparse
    I_ comp_nids_;
    V_ *vals_;  // used for CSR/CSC/CSB
    V_ *valsw_;
    N_ *offset_;
    CSRC_TYPE type_;
    MatrixProperties mtx_prop_;
    Alloc<I_> ids_alloc_;
    Alloc<V_> vals_alloc_;
    Alloc<N_> offset_alloc_;

    I_ *col_ids_;  // used for COO
    I_ *row_ids_;  // used for COO, and DCSR

    I_ *cperm_;
    I_ *ciperm_;

    I_ *rperm_;
    I_ *riperm_;

    bool *do_GEMM_work_;
};

#endif