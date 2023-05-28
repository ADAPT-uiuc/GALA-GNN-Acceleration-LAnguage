#ifndef _DENSE_MATRIX_H
#define _DENSE_MATRIX_H

#include "../utils/info_error.h"
#include "matrix_prop.h"
// #include "matrix.h"
// struct MatrixProperties;

template<typename I_, typename N_, typename V_, template<class A> class Alloc = std::allocator>
class DenseMatrix {
public:
    typedef V_ vtype;
    typedef I_ itype;
    typedef N_ ntype;

    enum class DENSE_MTX_TYPE {
        CM, RM
    };

    DenseMatrix() : nrows_(0), ncols_(0), nvals_(0), vals_(nullptr) {}

    ~DenseMatrix() { clear(); }

    // doesn't copy the arrays
    INFO import_mtx(I_ nrows, I_ ncols, N_ nvals, V_ *vals, DENSE_MTX_TYPE order,
                    MatrixProperties mtx_prop = MatrixProperties()) { return INFO::SUCCESS; }

    INFO import_mtx(I_ nrows, I_ ncols, N_ nvals, V_ *vals) {
        nrows_ = nrows;
        ncols_ = ncols;
        nvals_ = nvals;
        vals_ = vals;
        return INFO::SUCCESS;
    }

    INFO import_mtx(V_ *vals) {
        vals_ = vals;
        return INFO::SUCCESS;
    }

    INFO export_mtx(I_ &nrows, I_ &ncols, N_ &nvals, V_ *&vals, DENSE_MTX_TYPE order,
                    MatrixProperties &mtx_prop) { return INFO::SUCCESS; }

    // INFO build(I_ nrows, I_ ncols, N_ nvals, I_ * row_ids, I_ *col_ids, V_ *vals, MatrixProperties mtx_prop= MatrixProperties()) { return INFO::SUCCESS;}

    INFO build(I_ nrows, I_ ncols, V_ *vals, DENSE_MTX_TYPE type, MatrixProperties mtx_prop = MatrixProperties()) {
        nvals_ = (N_) (nrows) * (N_) (ncols);
        nrows_ = nrows;
        ncols_ = ncols;
        type_ = type;
        mtx_prop_ = mtx_prop;
        vals_ = vals_alloc_.allocate(nvals_ + 4 * ncols_);

        if (type == DENSE_MTX_TYPE::RM) {
#pragma omp parallel for
            for (N_ e = 0; e < nvals_; e++) {
//                std::cout << e << " " << vals[e] << " " << e << std::endl;
                vals_[e] = vals[e];
            }
        } else if (type == DENSE_MTX_TYPE::CM) {
#pragma omp parallel for
            for (N_ e = 0; e < nvals_; e++) {
                N_ indx = (e % ncols) * nrows + (e / ncols);
//                std::cout << e << " " << vals[e] << " " << indx <<std::endl;
                vals_[indx] = vals[e];
            }
        }

        return INFO::SUCCESS;
    }

    INFO build(I_ nrows, I_ ncols, V_ *vals, DENSE_MTX_TYPE type, size_t align, MatrixProperties mtx_prop = MatrixProperties()) {
        nvals_ = (N_) (nrows) * (N_) (ncols);
        nrows_ = nrows;
        ncols_ = ncols;
        type_ = type;
        mtx_prop_ = mtx_prop;
        if (align == 0){
            vals_ = (V_*)aligned_alloc(64, (nvals_ + 4 * ncols_)* sizeof(V_));
        } else {
            vals_ = (V_*)aligned_alloc(align, (nvals_ + 4 * ncols_)* sizeof(V_));
        }

        if (type == DENSE_MTX_TYPE::RM) {
#pragma omp parallel for
            for (N_ e = 0; e < nvals_; e++) {
//                std::cout << e << " " << vals[e] << " " << e << std::endl;
                vals_[e] = vals[e];
            }
        } else if (type == DENSE_MTX_TYPE::CM) {
#pragma omp parallel for
            for (N_ e = 0; e < nvals_; e++) {
                N_ indx = (e % ncols) * nrows + (e / ncols);
//                std::cout << e << " " << vals[e] << " " << indx <<std::endl;
                vals_[indx] = vals[e];
            }
        }

        return INFO::SUCCESS;
    }

    INFO clone_mtx(I_ nrows, I_ ncols, V_ *vals, DENSE_MTX_TYPE type, size_t align, MatrixProperties mtx_prop = MatrixProperties()) {
        nvals_ = (N_) (nrows) * (N_) (ncols);
        nrows_ = nrows;
        ncols_ = ncols;

        V_ *new_vals_ = (V_ *) aligned_alloc(64, (nvals_+ 4 * ncols_) * sizeof(V_));
        std::copy(vals, vals + nvals_, new_vals_);

        vals_ = new_vals_;

        mtx_prop_ = mtx_prop;
        type_ = type;
        return INFO::SUCCESS;
    }

    INFO build(I_ nrows, I_ ncols, DENSE_MTX_TYPE type, MatrixProperties mtx_prop = MatrixProperties()) {
        nvals_ = (N_) (nrows) * (N_) (ncols);
        nrows_ = nrows;
        ncols_ = ncols;
        type_ = type;
        //std::cout<<"Size of dense "<<nrows_<<" "<<ncols_<<" ==> "<<nvals_<<std::endl;
        vals_ = vals_alloc_.allocate(nvals_ + 4 * ncols_);

        return INFO::SUCCESS;
    }

    INFO build(I_ nrows, I_ ncols, DENSE_MTX_TYPE type, size_t align, MatrixProperties mtx_prop = MatrixProperties()) {
        nvals_ = (N_) (nrows * ncols);
        nrows_ = nrows;
        ncols_ = ncols;
        type_ = type;
        //std::cout<<"Size of dense "<<nrows_<<" "<<ncols_<<" ==> "<<nvals_<<std::endl;
        if (align == 0){
            vals_ = (V_*)aligned_alloc(64,(nvals_ + 4 * ncols_)* sizeof(V_));
        } else {
            vals_ = (V_*)aligned_alloc(align, (nvals_ + 4 * ncols_)* sizeof(V_));
        }

        return INFO::SUCCESS;
    }


    void set_all(V_ val) {
        if (vals_ == nullptr) {
            vals_ = (V_*)vals_alloc_.allocate(nvals_ + 4 * ncols_);
        }
#pragma omp parallel for
        for (N_ v = 0; v < nvals_; v++) { vals_[v] = val; }
    }

    void set_all(V_ val, size_t align) {
        if (vals_ == nullptr) {
            if (align == 0){
                vals_ = (V_*)aligned_alloc(64,(nvals_ + 4 * ncols_)* sizeof(V_));
            } else {
                vals_ = (V_*)aligned_alloc(align, (nvals_ + 4 * ncols_)* sizeof(V_));
            }
        }
#pragma omp parallel for
        for (N_ v = 0; v < nvals_; v++) { vals_[v] = val; }
    }

    void sum_all(V_ &val) {
        V_ sum = 0;
#pragma omp parallel for reduction(+:sum)
        for (N_ v = 0; v < nvals_; v++) { sum += vals_[v]; }
        val = sum;
    }

// #ifdef __INTEL_COMPILER
// get functions
    inline I_ nrows() const { return nrows_; }

    inline I_ ncols() const { return ncols_; }

    inline N_ nvals() const { return nvals_; }

    inline DENSE_MTX_TYPE type() const { return type_; }

    inline V_ *vals_ptr() const { return vals_; }

    INFO clear() {
        if (vals_ != nullptr) {
            vals_alloc_.deallocate(vals_, nvals_ + 4 * ncols_);
            vals_ = nullptr;
        }
        return INFO::SUCCESS;
    }

private:
    I_ nrows_;
    I_ ncols_;
    N_ nvals_;

//   N_ *offset_;

//     //   used for CSB and BCSR
//   I_ blk_dimr_;  // row dimension of single blk
//   I_ blk_dimc_;  // col dimension of single blk
//   I_ mtx_dimr_;  // number of blk in mtx in row dimension
//   I_ mtx_dimc_;  // number of blk in mtx in col dimension

    // arrays for regular nzlist representation
//   I_ *col_ids_;  // used for CSR
//   I_ *row_ids_;  // used for CSC
    V_ *vals_;     // used for CSR/CSC/CSB

//   bool is_compressed_;
    MatrixProperties mtx_prop_;
    DENSE_MTX_TYPE type_;
    Alloc<V_> vals_alloc_;

    // template <typename Ir_, typename Nr_, typename Vr_, template <class A> class Allocr>
    // friend class Matrix;
};

#endif