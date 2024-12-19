#ifndef _COO_MATRIX_H
#define _COO_MATRIX_H

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

template<typename I_, typename N_, typename V_, template<class A> class Alloc = std::allocator>
class COOMatrix {
public:
    typedef V_ vtype;
    typedef I_ itype;
    typedef N_ ntype;

    COOMatrix() : nrows_(0), ncols_(0), nvals_(0),
                  col_ids_(nullptr), row_ids_(nullptr), vals_(nullptr) {}

    ~COOMatrix() { clear(); }

    INFO
    build(I_ nrows, I_ ncols, N_ nvals, I_ *&row_ids, I_ *&col_ids, V_ *&vals, CSRC_TYPE type, int8_t is_sorted = -1,
          bool has_dup = true, MatrixProperties mtx_prop = MatrixProperties()) {
        type_ = type;
        nrows_ = nrows;
        ncols_ = ncols;
        nvals_ = nvals;
        mtx_prop_ = mtx_prop;
        type_ = type;

        if (vals != nullptr) vals_ = vals_alloc_.allocate(nvals_);
        row_ids_ = row_ids;
        col_ids_ = col_ids;

//        if (type_ == CSRC_TYPE::COO_CO) {
//            return make_col_ordered();
//        } else {
//            return make_row_ordered();
//        };
        return INFO::SUCCESS;
    }

    INFO make_col_ordered() {
        std::pair<I_, I_> temp_pair[nvals_];

        // Storing the respective array
        // elements in pairs.
#pragma omp parallel for
        for (int i = 0; i < nvals_; i++) {
            temp_pair[i].first = col_ids_[i];
            temp_pair[i].second = row_ids_[i];
        }

        // Sorting the pair array.
        sort(temp_pair, temp_pair + nvals_);

        // Modifying original arrays
#pragma omp parallel for
        for (int i = 0; i < nvals_; i++) {
            col_ids_[i] = temp_pair[i].first;
            row_ids_[i] = temp_pair[i].second;
        }

        return INFO::SUCCESS;
    }

    INFO make_row_ordered() {
        std::pair<I_, I_> temp_pair[nvals_];

        // Storing the respective array
        // elements in pairs.
#pragma omp parallel for
        for (int i = 0; i < nvals_; i++) {
            temp_pair[i].first = row_ids_[i];
            temp_pair[i].second = col_ids_[i];
        }

        // Sorting the pair array.
        sort(temp_pair, temp_pair + nvals_);

        // Modifying original arrays
#pragma omp parallel for
        for (int i = 0; i < nvals_; i++) {
            row_ids_[i] = temp_pair[i].first;
            col_ids_[i] = temp_pair[i].second;
        }

        return INFO::SUCCESS;
    }

    INFO clear() {
        if (col_ids_ != nullptr) {
            ids_alloc_.deallocate(col_ids_, nvals_);
            col_ids_ = nullptr;
        }
        if (row_ids_ != nullptr) {
            ids_alloc_.deallocate(row_ids_, nvals_);
            row_ids_ = nullptr;
        }
        if (vals_ != nullptr) {
            vals_alloc_.deallocate(vals_, nvals_);
            vals_ = nullptr;
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

    inline I_ *col_ids_ptr() const { return col_ids_; }

    inline I_ *row_ids_ptr() const { return row_ids_; }

    inline V_ *vals_ptr() const { return vals_; }

    inline void set_mtx_name(string mtx_name) { mtx_name_ = mtx_name; }

    inline string mtx_name() const { return mtx_name_; }

    // For IDs and Offsets of CSR based ones

    inline N_ *offset_ptr() const { return offset_; }

    inline I_ *ids_ptr() const { return ids_; }

private:
    string mtx_name_;
    // MtxStats mtx_stats_;
    I_ nrows_;
    I_ ncols_;
    N_ nvals_;
    I_ *col_ids_;  // used for COO
    I_ *row_ids_;  // used for COO
    V_ *vals_;  // used for COO/CSR/CSC/CSB

    // Used for the CSR based ones
    I_ *ids_;  // used for CSR
    N_ *offset_;

    CSRC_TYPE type_;
    MatrixProperties mtx_prop_;
    Alloc<I_> ids_alloc_;
    Alloc<V_> vals_alloc_;
    Alloc<N_> offset_alloc_;
};

#endif