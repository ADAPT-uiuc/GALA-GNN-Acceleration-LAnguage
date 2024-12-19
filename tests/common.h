//
// Created by damitha on 6/6/22.
//
#include "../src/utils/mtx_io.h"
#include "../src/utils/threading_utils.h"

#include "../src/formats/csrc_matrix.h"
#include "../src/formats/coo_matrix.h"

#include "../src/ops/aggregators.h"

#include "../src/third_party/libnpy/npy.hpp"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <random>
#include <omp.h>


template<class SM, class DM>
void getMaskSubgraphs(SM* adj, DM* mask, int layers, std::vector<SM *> &forward_vec, std::vector<SM *> &backward_vec){
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::vtype dvT;

    // iterate through all rows
    DM* cur_mask = mask;
    iT nrows = adj->nrows();
    iT ncols = adj->ncols();
    nT src_nvals = adj->nvals();
    iT *src_ids = adj->ids_ptr();
    nT *src_offset = adj->offset_ptr();
    vT *src_vals = adj->vals_ptr();

    for (int l = 0; l < layers; l++){
        nT *new_offset = (nT *) aligned_alloc(64, (nrows + 1) * sizeof(nT));
        std::fill(new_offset, new_offset + (nrows + 1), 0);
        // Get offsets added by each row

        dvT* mask_ptr = cur_mask->vals_ptr();

#pragma omp parallel for schedule(dynamic, 1)
        for (iT i = 0; i < nrows; i++){
            // new_offset[i + 1] = src_offset[i+1] - src_offset[i];
            if (mask_ptr[i] > 0){
                new_offset[i + 1] = src_offset[i+1] - src_offset[i];
            } else {
                new_offset[i + 1] = 0;
            }
        }

        for (iT i = 0; i < nrows; i++){
            new_offset[i + 1] = new_offset[i + 1] + new_offset[i];
        }

        nT new_nvals = new_offset[nrows];
        // std::cout << "nvals:" << new_nvals << std::endl;


        iT *new_ids = (iT *) aligned_alloc(64, (new_nvals) * sizeof(iT));
        vT *new_vals = (vT *) aligned_alloc(64, (new_nvals) * sizeof(vT));

#pragma omp parallel for schedule(dynamic, 1)
        for (iT i = 0; i < nrows; i++){
            nT local_e_start = new_offset[i];
            nT local_e_end = new_offset[i + 1];

            nT src_e_start = src_offset[i];
            nT src_e_end = src_offset[i + 1];

            // std::cout << "i: " << i << std::endl;
            // std::cout << "i1: " << local_e_start << " " << local_e_end << std::endl;
            // std::cout << "i2: " << src_e_start << " " << src_e_end << std::endl;

            for (nT j = 0; j < (local_e_end - local_e_start); j++){
                new_ids[local_e_start + j] = src_ids[src_e_start + j];
                new_vals[local_e_start + j] = src_vals[src_e_start + j];
            }

        }

        SM *new_adj = new SM();
        new_adj->import_csr(nrows,
                    ncols,
                    new_nvals,
                    new_ids,
                    new_vals,
                    new_offset);

        forward_vec.push_back(new_adj);

        SM *new_trans = new SM();
        buildTranspose(new_adj, new_trans);
        backward_vec.push_back(new_trans);

        auto maxaggr = maxAgg<dvT, dvT, nT>;
        DM* new_mask = new DM();
        new_mask->build(nrows, 1, cur_mask->type(), 0);
        // new_mask->clone_mtx(nrows, 1, cur_mask->vals_ptr(), mask->type(), 0);
        gSpMM(adj, cur_mask, new_mask, maxaggr);
        cur_mask = new_mask;
    }
}

template<class SM>
void buildTranspose(SM* src, SM* res){
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT; 
    typedef typename SM::vtype vT;

    // iT* sparse_ids = (iT *) aligned_alloc(64, (src->nvals()) * sizeof(iT));
    // iT* sparse_ids = src->ids_ptr();
    iT* sparse_ids = src->get_sids();
    iT* col_ids = src->ids_ptr();
    // iT* col_ids = (iT *) aligned_alloc(64, (src->nvals()) * sizeof(iT));
    vT* vals = src->vals_ptr();
    // vT* vals = (vT *) aligned_alloc(64, (src->nvals()) * sizeof(vT));

    // std::cout << src->ncols() << " " << src->nrows() << " " << src->nvals() << " " << col_ids[src->nvals() - 1] << " " << sparse_ids[src->nvals() - 1] << std::endl;
    res->build(src->ncols(), src->nrows(), src->nvals(), col_ids, sparse_ids, vals, CSRC_TYPE::CSR);
}

template<class DM1, class DM2>
void repopulate(DM1 *src, DM2 *dst){
    typedef typename DM1::itype iT;
    typedef typename DM1::ntype nT;
    // source / target types
    typedef typename DM1::vtype sT;
    typedef typename DM2::vtype tT;

    dst->build(src->nrows(), src->ncols(), DM2::DENSE_MTX_TYPE::RM, 0);
    
#pragma omp parallel for schedule(static)
    for (iT i = 0; i < src->nrows() * src->ncols(); i++) {
        dst->vals_ptr()[i] = (tT)src->vals_ptr()[i];
    }
}

template<class DM>
void readDM(std::string filename, DM *mtx, typename DM::DENSE_MTX_TYPE type) {
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    diT nrows, ncols;
    dvT *vals;

#ifdef RNPY
    std::vector<unsigned long> shape{};
    bool fortran_order;
    std::vector<dvT> data;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, data);

    nrows = (diT) shape.at(0);
    ncols = (diT) shape.at(1);

//    std::cout << filename << " | shape: " << nrows << " " << ncols << std::endl;

    vals = (dvT *) aligned_alloc(64, (nrows * ncols) * sizeof(dvT));
    std::copy(data.begin(), data.end(), vals);
#else
    MtxIO<diT, dnT, dvT> reader;
    reader.readMtx(filename);
    dnT nvals;
    dnT size;
    diT *col_ids, *row_ids;
    reader.getData(nrows, ncols, nvals, size, row_ids, col_ids, vals);
#endif

#ifdef A_ALLOC
    mtx->build(nrows, ncols, vals, type, 0);
#else
    mtx->build(nrows, ncols, vals, type);
#endif
}

template<class DM>
void readDV(std::string filename, DM *mtx, typename DM::DENSE_MTX_TYPE type) {
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    diT nrows, ncols;
    dvT *vals;

#ifdef RNPY
    std::vector<unsigned long> shape{};
    bool fortran_order;
    std::vector<dvT> data;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, data);

    nrows = (diT) shape.at(0);
    ncols = 1;

//    std::cout << filename << " | shape: " << nrows << " " << ncols << std::endl;

    vals = (dvT *) aligned_alloc(64, (nrows * ncols) * sizeof(dvT));
    std::copy(data.begin(), data.end(), vals);
#else
    MtxIO<diT, dnT, dvT> reader;
    reader.readMtx(filename);
    dnT nvals;
    dnT size;
    diT *col_ids, *row_ids;
    reader.getData(nrows, ncols, nvals, size, row_ids, col_ids, vals);
#endif

#ifdef A_ALLOC
    mtx->build(nrows, ncols, vals, type, 0);
#else
    mtx->build(nrows, ncols, vals, type);
#endif
}

template<class DM>
void readDM_d(std::string filename, DM *mtx, typename DM::DENSE_MTX_TYPE type) {
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    diT nrows, ncols;
    dvT *vals;

#ifdef RNPY
    std::vector<unsigned long> shape{};
    bool fortran_order;
    std::vector<double> data;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, data);

    nrows = (diT) shape.at(0);
    ncols = (diT) shape.at(1);

//    std::cout << filename << " | shape: " << nrows << " " << ncols << std::endl;

    vals = (dvT *) aligned_alloc(64, (nrows * ncols) * sizeof(dvT));
    std::copy(data.begin(), data.end(), vals);
#else
    MtxIO<diT, dnT, dvT> reader;
    reader.readMtx(filename);
    dnT nvals;
    dnT size;
    diT *col_ids, *row_ids;
    reader.getData(nrows, ncols, nvals, size, row_ids, col_ids, vals);
#endif

#ifdef A_ALLOC
    mtx->build(nrows, ncols, vals, type, 0);
#else
    mtx->build(nrows, ncols, vals, type);
#endif
}

template<class DM>
void readDM_dV(std::string filename, DM *mtx, typename DM::DENSE_MTX_TYPE type) {
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    diT nrows, ncols;
    dvT *vals;

#ifdef RNPY
    std::vector<unsigned long> shape{};
    bool fortran_order;
    std::vector<double> data;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, data);

    nrows = (diT) shape.at(0);
    ncols = 1;

//    std::cout << filename << " | shape: " << nrows << " " << ncols << std::endl;

    vals = (dvT *) aligned_alloc(64, (nrows * ncols) * sizeof(dvT));
    std::copy(data.begin(), data.end(), vals);
#else
    MtxIO<diT, dnT, dvT> reader;
    reader.readMtx(filename);
    dnT nvals;
    dnT size;
    diT *col_ids, *row_ids;
    reader.getData(nrows, ncols, nvals, size, row_ids, col_ids, vals);
#endif

#ifdef A_ALLOC
    mtx->build(nrows, ncols, vals, type, 0);
#else
    mtx->build(nrows, ncols, vals, type);
#endif
}

template<class SM>
void readSM_npy(std::string path, SM *adj) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    std::string filename;
    std::vector<unsigned long> shape{};
    bool fortran_order;

//    std::cout << 1 << std::endl;
    iT adj_nrows, adj_ncols;
    filename = path + "Adj_src.npy";
    shape.clear();
    std::vector<iT> data_adj_src;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, data_adj_src);
    adj_nrows = data_adj_src.at(0);
    adj_ncols = data_adj_src.at(1);
//    std::cout << 2 << std::endl;
    filename = path + "Adj_dst.npy";
    shape.clear();
    std::vector<iT> data_adj_dst;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, data_adj_dst);
    nT adj_nvals = (nT) shape.at(0);
    iT *adj_row_ids = (iT *) aligned_alloc(64, (adj_nvals) * sizeof(iT));
    std::copy(data_adj_src.begin() + 2, data_adj_src.end(), adj_row_ids);
    iT *adj_col_ids = (iT *) aligned_alloc(64, (adj_nvals) * sizeof(iT));
    std::copy(data_adj_dst.begin(), data_adj_dst.end(), adj_col_ids);
    vT *adj_vals = (vT *) aligned_alloc(64, (adj_nvals) * sizeof(vT));
//    std::cout << 3 << std::endl;
//    std::memset(adj_vals, 1, adj_nvals);

    adj->build(adj_nrows, adj_ncols, adj_nvals, adj_row_ids, adj_col_ids, adj_vals, CSRC_TYPE::CSR);
    adj->set_all(1);
    data_adj_dst.clear();
    data_adj_src.clear();
}

template<class SM>
void readSM_npy32(std::string path, SM *adj) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    std::string filename;
    std::vector<unsigned long> shape{};
    bool fortran_order;

    iT adj_nrows, adj_ncols;
    filename = path + "Adj_src.npy";
    shape.clear();
    std::vector<uint32_t> data_adj_src;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, data_adj_src);
    adj_nrows = (iT) data_adj_src.at(0);
    adj_ncols = (iT) data_adj_src.at(1);

    filename = path + "Adj_dst.npy";
    shape.clear();
    std::vector<uint32_t> data_adj_dst;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, data_adj_dst);
    nT adj_nvals = (nT) shape.at(0);
    iT *adj_row_ids = (iT *) aligned_alloc(64, (adj_nvals) * sizeof(iT));
    std::copy(data_adj_src.begin() + 2, data_adj_src.end(), adj_row_ids);
    iT *adj_col_ids = (iT *) aligned_alloc(64, (adj_nvals) * sizeof(iT));
    std::copy(data_adj_dst.begin(), data_adj_dst.end(), adj_col_ids);
    vT *adj_vals = (vT *) aligned_alloc(64, (adj_nvals) * sizeof(vT));
//    std::memset(adj_vals, 1, adj_nvals);

    // adj->build(adj_nrows, adj_ncols, adj_nvals, adj_col_ids, adj_row_ids, adj_vals, CSRC_TYPE::CSR);
    adj->build(adj_nrows, adj_ncols, adj_nvals, adj_row_ids, adj_col_ids, adj_vals, CSRC_TYPE::CSR);
    adj->set_all(1);
    data_adj_dst.clear();
    data_adj_src.clear();
}

template<class DM>
void readDM_npy(std::string filename, DM *mtx, typename DM::DENSE_MTX_TYPE type) {
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    diT nrows, ncols;
    dvT *vals;

    std::vector<unsigned long> shape{};
    bool fortran_order;
    std::vector<dvT> data;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, data);

    nrows = (diT) shape.at(0);
    ncols = (diT) shape.at(1);

    vals = (dvT *) aligned_alloc(64, (nrows * ncols) * sizeof(dvT));
    std::copy(data.begin(), data.end(), vals);

    mtx->build(nrows, ncols, vals, type, 0);
}

template<class SM>
void readSM(std::string filename, CSRCMatrix<typename SM::itype, typename SM::ntype, typename SM::vtype> *mtx) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    MtxIO<iT, nT, vT> reader;
    reader.readMtx(filename);
    iT nrows, ncols;
    nT nvals;
    nT size;
    iT *col_ids, *row_ids;
    vT *vals;
    reader.getData(nrows, ncols, nvals, size, row_ids, col_ids, vals);

//    for (iT i = 0; i<nvals; i++){
//        std::cout << row_ids[i] + 1 << " " << col_ids[i] + 1 << std::endl;
//    }

    mtx->build(nrows, ncols, nvals, row_ids, col_ids, vals, CSRC_TYPE::CSR);
}

template<class SM>
void readSM(std::string filename, COOMatrix<typename SM::itype, typename SM::ntype, typename SM::vtype> *mtx,
            CSRC_TYPE type) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;


    MtxIO<iT, nT, vT> reader;
    reader.readMtx(filename);
    iT nrows, ncols;
    nT nvals;
    nT size;
    iT *col_ids, *row_ids;
    vT *vals;
    reader.getData(nrows, ncols, nvals, size, row_ids, col_ids, vals);
//    std::cout << row_ids[0] << " " << col_ids[0] << std::endl;
//    std::cout << row_ids[1] << " " << col_ids[1] << std::endl;
//    std::cout << row_ids[2] << " " << col_ids[2] << std::endl;
    mtx->build(nrows, ncols, nvals, row_ids, col_ids, vals, type);
//    std::cout << "AAA" << std::endl;
//    std::cout << row_ids[0] << " " << col_ids[0] << std::endl;
//    std::cout << row_ids[1] << " " << col_ids[1] << std::endl;
//    std::cout << row_ids[2] << " " << col_ids[2] << std::endl;
//    std::cout << "AAA" << std::endl;
//    std::cout << mtx->row_ids_ptr()[0] << " " << mtx->col_ids_ptr()[0] << std::endl;
//    std::cout << mtx->row_ids_ptr()[1] << " " << mtx->col_ids_ptr()[1] << std::endl;
//    std::cout << mtx->row_ids_ptr()[2] << " " << mtx->col_ids_ptr()[2] << std::endl;
}

template<class SM>
void writeSM_mtx(SM *adj, std::string &out_path) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    // Create and open file
    std::ofstream out_mtx(out_path);
    // Identifier line
    out_mtx << "%%MatrixMarket matrix coordinate real general\n";

    iT adj_nrows = adj->nrows();
    iT adj_ncols = adj->ncols();
    iT adj_nvals = adj->nvals();


    vT *adj_vals_ptr = adj->vals_ptr();
    nT *adj_offset_ptr = adj->offset_ptr();
    iT *adj_ids_ptr = adj->ids_ptr();

    out_mtx << adj_nrows << " " << adj_ncols << " " << adj_nvals << "\n";
    // TODO remember that this iT index is 1 less than the index for the mtx file.
    //  i.e. you need to add 1
    for (iT v = 0; v < adj_nrows; v++) {
        for (nT e = adj_offset_ptr[v]; e < adj_offset_ptr[v + 1]; e++) {
            iT u = adj_ids_ptr[e];
            vT val = adj_vals_ptr[e];
            out_mtx << v + 1 << " " << u + 1 << " " << val << "\n";
        }
    }

    // Close the file
    out_mtx.close();
}

template<class SM>
void
get_row_ids(CSRCMatrix<typename SM::itype, typename SM::ntype, typename SM::vtype> *mtx, typename SM::itype *&row_ids) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    row_ids = (iT *) aligned_alloc(64, mtx->nvals() * sizeof(iT));
#pragma omp parallel for schedule(static)
    for (iT i = 0; i < mtx->nrows(); i++) {
        nT start_row = mtx->offset_ptr()[i];
        nT end_row = mtx->offset_ptr()[i + 1];
        for (nT j = start_row; j < end_row; j++) {
            row_ids[j] = i;
        }
    }
}

#ifdef ACC_COLD
#pragma optimize("", off)
float cahce_flush(long val_i, long val_j){
    // Init data
    auto *X = (double*)aligned_alloc(64, val_i*val_j*sizeof(double));
    auto *Y = (double*)aligned_alloc(64, val_i*val_j*sizeof(double));

    // Random number generator C++11
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<long> dis;

    long r_num = dis(gen) % 10 + 1;
    long r_mod = dis(gen) % 5 + 1;
    for (int x = 0; x < 3; x++){
#pragma omp parallel for schedule(dynamic)
        for(long r_i=0; r_i<val_i; r_i++) {
            for(long r_j=0; r_j<val_j; r_j++) {
                Y[r_i*val_j + r_j] = (r_i * r_num) % r_mod;
                X[r_i*val_j + r_j] = Y[r_i*val_j + r_j] / r_num;
            }
        }
    }
    long rand_ind = (rand()%val_i*val_j);
    double ret = X[rand_ind] + Y[rand_ind];
    free(X);
    free(Y);
    return ret;
}
#pragma optimize("", on)
#endif

template<class SM>
void check_equal(SM *adj1, SM *adj2) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    bool is_diff = false;
    if (adj1->nrows() != adj2->nrows()) {
        is_diff = true;
    }
    if (is_diff) {
        std::cout << "The number of rows are different." << std::endl;
    }
    if (!is_diff) {
        for (iT i = 0; i < adj1->nrows() + 1; i++) {
            if (adj1->offset_ptr()[i] != adj2->offset_ptr()[i]) {
                is_diff = true;
                break;
            }
        }
    }
    if (is_diff) {
        std::cout << "The offsets are different." << std::endl;
    }

    if (!is_diff) {
        for (nT j = 0; j < adj1->nvals(); j++) {
            if (adj1->vals_ptr()[j] != adj2->vals_ptr()[j]) {
                is_diff = true;
                break;
            }
        }
    }
    if (is_diff) {
        std::cout << "The values are different." << std::endl;
    }

    if (!is_diff) {
        for (nT j = 0; j < adj1->nvals(); j++) {
            if (adj1->ids_ptr()[j] != adj2->ids_ptr()[j]) {
                is_diff = true;
                break;
            }
        }
    }
    if (is_diff) {
        std::cout << "The ids are different." << std::endl;
    }

    if (is_diff) {
        std::cout << "The graphs are different." << std::endl;
    } else {
        std::cout << "The graphs are the same." << std::endl;
    }
}

std::tuple<double, double> calc_mean_std(std::vector<double> vec) {
    if ((double) vec.size() > 1) {
        double mean = 0;
        for (auto val: vec) {
            mean += val;
        }
        mean /= (double) vec.size();

        double std = 0;
        std::for_each(std::begin(vec), std::end(vec), [&](const double d) {
            std += (d - mean) * (d - mean);
        });
        return std::make_tuple(mean, sqrt(std / ((double) vec.size() - 1)));
    } else if ((double) vec.size() > 0) {
        return std::make_tuple(vec.at(0), 0);
    } else {
        return std::make_tuple(0, 0);
    }
}

double calc_std(std::vector<double> &vec) {
    if (vec.size() > 1) {
        double mean = 0;
        for (auto val: vec) {
            mean += val;
        }
        mean /= (double) vec.size();

        double std = 0;
        std::for_each(std::begin(vec), std::end(vec), [&](const double d) {
            std += (d - mean) * (d - mean);
        });
        return (std / ((double) vec.size() - 1));
    }
    return 0;
}

double calc_mean(std::vector<double> &vec) {
    if ((double) vec.size() > 0) {
        double mean = 0;
        for (auto val: vec) {
            mean += val;
        }
        mean /= (double) vec.size();

        return mean;
    }
    return 0;
}

//template<class SM>
//std::vector<edge_list::aux::edge> readNReorderSM(std::string filename) {
//    typedef typename SM::itype iT;
//    typedef typename SM::ntype nT;
//    typedef typename SM::vtype vT;
//
//
//    MtxIO<iT, nT, vT> reader;
//    reader.readMtx(filename);
//    iT nrows, ncols;
//    nT nvals;
//    nT size;
//    iT *col_ids, *row_ids;
//    vT *vals;
//    reader.getData(nrows, ncols, nvals, size, row_ids, col_ids, vals);
//
//    std::vector<edge_list::aux::edge> edges;
//    edges.resize(nvals);
//
//    double work_thread = (double) nvals / PTHHREADS;
//    if (work_thread < 1) {
//        work_thread = 1;
//    }
//
//    auto edges_start = edges.begin();
//#pragma omp parallel for schedule(static)
//    for (int i = 0; i < PTHHREADS; i++) {
//        for (nT j = std::ceil(work_thread * i); j < std::ceil(work_thread * (i + 1)); j++) {
//
//            edges[j] = std::make_tuple((rabbit_order::vint) row_ids[j], (rabbit_order::vint) col_ids[j],
//                                       (float) vals[j]);
//        }
//
//    }
//    return edges;
//}

//template<class SM>
//std::vector<edge_list::aux::edge> reorderSM(typename SM::ntype &nvals,
//                                            typename SM::itype *&row_ids,
//                                            typename SM::itype *&col_ids,
//                                            typename SM::vtype *&vals) {
//    typedef typename SM::itype iT;
//    typedef typename SM::ntype nT;
//    typedef typename SM::vtype vT;
//
//    std::vector<edge_list::aux::edge> edges;
//    edges.resize(nvals);
//
//    double work_thread = (double) nvals / PTHHREADS;
//    if (work_thread < 1) {
//        work_thread = 1;
//    }
//
//    auto edges_start = edges.begin();
//#pragma omp parallel for schedule(static)
//    for (int i = 0; i < PTHHREADS; i++) {
//        for (nT j = std::ceil(work_thread * i); j < std::ceil(work_thread * (i + 1)); j++) {
//
//            edges[j] = std::make_tuple((rabbit_order::vint) row_ids[j], (rabbit_order::vint) col_ids[j],
//                                       (float) vals[j]);
//        }
//
//    }
//    return edges;
//}