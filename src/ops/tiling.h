//
// Created by damitha on 2/18/22.
//

#ifndef SPARSE_ACCELERATOR_TILING_H
#define SPARSE_ACCELERATOR_TILING_H

// TODO -  This WILL work with CSR, but not sure of others at the moment
// Tiling for Sparse + Dense matrix operations
template<class SM, class DM>
struct GNOpTile {
protected:
    // Sparse input (Graph)
    typedef typename SM::itype iT;

    // DM Dense input / output (NN)
    typedef typename DM::itype diT;

public:
    // Rows to get from the sparse matrix
    iT srows_start;
    iT srows_end; // Not inclusive
    // Rows to get from the dense matrix
#ifndef ST_0
    diT drows_start;
    diT drows_end; // Not inclusive
#endif
//    // Going to assume for now that if dense matrices are partitioned then they would have to be made into contiguous
//    // memory locations. (i.e. Commenting the code below)
//    // Columns to get from the dense matrix (Number of features)
//    dnT dcols_start;
//    dnT dcols_end;
};

template<class SM>
void check_equal_tile(SM *adj1, SM *adj2) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    bool is_diff = false;
    bool any_diff = false;
    if (adj1->nrows() != adj2->nrows()) {
        is_diff = true;
        any_diff = true;
        std::cout << "A: " << adj1->nrows() << " B: " << adj2->nrows() << std::endl;
    }
    if (is_diff) {
        std::cout << "The number of rows are different." << std::endl;
    }

    is_diff = false;
    if (!is_diff) {
        for (iT i = 0; i < adj1->nrows() + 1; i++) {
            if (adj1->offset_ptr()[i] != adj2->offset_ptr()[i]) {
                is_diff = true;
                any_diff = true;
                break;
            }
        }
    }
    if (is_diff) {
        std::cout << "The offsets are different." << std::endl;
    }

    is_diff = false;
    if (!is_diff) {
        for (nT j = 0; j < adj1->nvals(); j++) {
            if (adj1->vals_ptr()[j] != adj2->vals_ptr()[j]) {
                is_diff = true;
                any_diff = true;
                break;
            }
        }
    }
    if (is_diff) {
        std::cout << "The values are different." << std::endl;
    }

    is_diff = false;
    if (!is_diff) {
        for (nT j = 0; j < adj1->nvals(); j++) {
            if (adj1->ids_ptr()[j] != adj2->ids_ptr()[j]) {
                is_diff = true;
                any_diff = true;
                break;
            }
        }
    }
    if (is_diff) {
        std::cout << "The ids are different." << std::endl;
    }

    if (!any_diff) {
        std::cout << "The graphs are the same." << std::endl;
    }
}

template<class LDM>
std::vector<LDM *> static_tiling_numfeats(LDM *src, typename LDM::ntype num_feats) {
    // Get types
    typedef typename LDM::itype ldiT;
    typedef typename LDM::ntype ldnT;
    typedef typename LDM::vtype ldvT;

    // Get stats from the original matrox
    ldiT src_ncols = src->ncols();
    ldiT src_nrows = src->nrows();
    ldvT *src_vals_ptr = src->vals_ptr();

    // Resulting vector
    std::vector<LDM *> res;

#pragma omp parallel for
    for (ldiT j_start = 0; j_start < src_ncols; j_start += num_feats) {
        // Calculate result's column numbers / indexes
        ldiT j_sum_end = j_start + num_feats;
        ldiT j_end = std::min(j_sum_end, src_ncols);
        ldiT j_diff = j_end - j_start;

        // Init new dense matrix and get stats
        LDM *new_dense = new LDM;
        new_dense->build(src_nrows,
                         j_diff,
                         src->type());
        ldvT *new_vals_ptr = new_dense->vals_ptr();
#pragma omp parallel for
        for (ldiT i_i = 0; i_i < src_nrows; i_i++) {
            // Calculate offsets for the row
            ldnT src_offset = i_i * src_ncols + j_start;
            ldnT new_offset = i_i * j_diff;
            // Assign values
            for (ldiT i_j = 0; i_j < j_diff; i_j++) {
                new_vals_ptr[new_offset + i_j] = src_vals_ptr[src_offset + i_j];
            }
        }
        res.push_back(new_dense);
    }
    return res;
}

// Need to produce this for BOTH input and output
template<class SM>
std::vector<SM *> tiling_adj_rows(SM *src,
                                  typename SM::itype ntiles,
                                  typename SM::itype *toffsets) {
    // Resulting vector
    std::vector<SM *> res;

    // Get types
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    omp_lock_t writelock;
    omp_init_lock(&writelock);

    // Get stats from the original matrox
    iT src_ncols = src->ncols();
    iT src_nrows = src->nrows();

    nT *src_offset_ptr = src->offset_ptr();
    iT *src_ids_ptr = src->ids_ptr();
    vT *src_vals_ptr = src->vals_ptr();

#pragma omp parallel for
    for (iT i = 0; i < ntiles; i++) {
        iT i_start = toffsets[i];
        iT i_end = toffsets[i + 1];

        SM *new_adj = new SM();

        nT new_nvals = 0;
        std::vector<nT> new_offset_ptr_vec;
        std::vector<iT> new_ids_ptr_vec;
        std::vector<vT> new_vals_vec;

        new_offset_ptr_vec.push_back(0);

        for (iT i_i = i_start; i_i < i_end; i_i += 1) {
            nT first_node_edge = src_offset_ptr[i_i];
            nT last_node_edge = src_offset_ptr[i_i + 1];
            for (nT e = first_node_edge; e < last_node_edge; e++) {
                iT u = src_ids_ptr[e];
                vT val = src_vals_ptr[e];

                new_ids_ptr_vec.push_back(u);
                new_vals_vec.push_back(val);

                new_nvals += 1;
            }
            new_offset_ptr_vec.push_back(new_nvals);
        }

        nT *new_offset_ptr = (nT *) malloc((i_end - i_start + 1) * sizeof(nT));
        std::copy(new_offset_ptr_vec.begin(), new_offset_ptr_vec.end(), new_offset_ptr);
        iT *new_ids_ptr = (iT *) malloc((new_nvals) * sizeof(iT));
        std::copy(new_ids_ptr_vec.begin(), new_ids_ptr_vec.end(), new_ids_ptr);
        vT *new_vals = (vT *) malloc((new_nvals) * sizeof(vT));
        std::copy(new_vals_vec.begin(), new_vals_vec.end(), new_vals);

        new_adj->import_csr(i_end - i_start,
                            src_ncols,
                            new_nvals,
                            new_ids_ptr,
                            new_vals,
                            new_offset_ptr);

#pragma omp critical
        {
            res.push_back(new_adj);
        }
    }
    return res;
}

#ifdef PT_0
// Need to produce this for BOTH input and output
// TODO could make an unordered version of this? i.e. running parallely and adding as completed
template<class SM>
void ord_col_tiling(std::vector<typename SM::itype> &col_breakpoints,
                    std::vector<SM *> &res,
                    typename SM::itype current_position) {
    // Get types
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    // Matrix tiled
    SM *src = res.at(current_position);

    std::vector<SM *> temp_res;

    // Get stats from the original matrox
    iT src_ncols = src->ncols();
    iT src_nrows = src->nrows();

    nT *src_offset_ptr = src->offset_ptr();
    iT *src_ids_ptr = src->ids_ptr();
    vT *src_vals_ptr = src->vals_ptr();

    auto copy_offsets = (nT *) aligned_alloc(64, (src->nrows() + 1) * sizeof(nT));
    memcpy(copy_offsets, src->offset_ptr(), (src->nrows() + 1) * sizeof(nT));

    // TODO might be able to parallelize this section if you try to find the values per row?
    //  But would still have an issue if col seg is empty. => Do DCSC?
    for (iT nth_tile = 0; nth_tile < col_breakpoints.size() - 1; nth_tile++) {
        iT j_start = col_breakpoints.at(nth_tile);
        iT j_end = col_breakpoints.at(nth_tile + 1);

        SM *new_adj = new SM();

        nT new_nvals = 0;
        std::vector<nT> new_offset_ptr_vec;
        std::vector<iT> new_ids_ptr_vec;
        std::vector<vT> new_vals_vec;

#ifdef CCMP
        iT new_nrows = 0;
        std::vector<iT> new_rows_vec;
#endif

        new_offset_ptr_vec.push_back(0);
        for (iT i_i = 0; i_i < src_nrows; i_i += 1) {
            nT first_node_edge = copy_offsets[i_i];
            nT last_node_edge = src_offset_ptr[i_i + 1];

#ifdef CCMP
            bool found_nnz = false;
#endif
            for (nT e = first_node_edge; e < last_node_edge; e++) {
                iT u = src_ids_ptr[e];
                if (u >= j_start && u < j_end) {
                    vT val = src_vals_ptr[e];
#ifdef CCMP
                    found_nnz = true;
#endif
                    new_ids_ptr_vec.push_back(u);
                    new_vals_vec.push_back(val);

                    new_nvals += 1;
                } else if (u >= j_end) {
                    copy_offsets[i_i] = e;
                    break;
                }
            }
#ifdef CCMP
            if (found_nnz) {
                new_offset_ptr_vec.push_back(new_nvals);
                new_rows_vec.push_back(i_i);
                new_nrows += 1;
            }
#else
            new_offset_ptr_vec.push_back(new_nvals);
#endif
        }
#ifdef A_ALLOC
#ifdef CCMP
        nT *new_offset_ptr = (nT *) aligned_alloc(64, (new_nrows + 1) * sizeof(nT));
#else
        nT *new_offset_ptr = (nT *) aligned_alloc(64, (src_nrows + 1) * sizeof(nT));
#endif
#else
#ifdef CCMP
        nT *new_offset_ptr = (nT *) malloc((val_rows + 1) * sizeof(nT));
#else
        nT *new_offset_ptr = (nT *) malloc((src_nrows + 1) * sizeof(nT));
#endif
        nT *new_offset_ptr = (nT *) malloc((src_nrows + 1) * sizeof(nT));
#endif
        std::copy(new_offset_ptr_vec.begin(), new_offset_ptr_vec.end(), new_offset_ptr);


#ifdef A_ALLOC
        iT *new_ids_ptr = (iT *) aligned_alloc(64, (new_nvals) * sizeof(iT));
#else
        iT *new_ids_ptr = (iT *) malloc((new_nvals) * sizeof(iT));
#endif
        std::copy(new_ids_ptr_vec.begin(), new_ids_ptr_vec.end(), new_ids_ptr);

#ifdef A_ALLOC
        vT *new_vals = (vT *) aligned_alloc(64, (new_nvals) * sizeof(vT));
#else
        vT *new_vals = (vT *) malloc((new_nvals) * sizeof(vT));
#endif
        std::copy(new_vals_vec.begin(), new_vals_vec.end(), new_vals);

#ifdef CCMP
#ifdef A_ALLOC
        iT *new_rows = (iT *) aligned_alloc(64, (new_nrows) * sizeof(iT));
#else
        iT *new_rows = (iT *) malloc((val_rows) * sizeof(iT));
#endif
        std::copy(new_rows_vec.begin(), new_rows_vec.end(), new_rows);
        new_adj->import_dcsr(new_nrows,
                             j_end - j_start,
                             new_nvals,
                             new_ids_ptr,
                             new_vals,
                             new_offset_ptr,
                             new_rows);

        new_rows_vec.clear();
        new_rows_vec.shrink_to_fit();
#else
        new_adj->import_csr(src_nrows,
                            j_end - j_start,
                            new_nvals,
                            new_ids_ptr,
                            new_vals,
                            new_offset_ptr);
#endif

        temp_res.push_back(new_adj);

        new_offset_ptr_vec.clear();
        new_offset_ptr_vec.shrink_to_fit();
        new_ids_ptr_vec.clear();
        new_ids_ptr_vec.shrink_to_fit();
        new_vals_vec.clear();
        new_vals_vec.shrink_to_fit();

//        std::cout << "A:  " << new_offset_ptr_vec.size() << " " << new_offset_ptr_vec.capacity() << std::endl;
    }

    auto insert_position = res.begin() + current_position;
    res.erase(insert_position);

    for (auto tiled_SM: temp_res) {
        insert_position = res.insert(insert_position, tiled_SM);
        insert_position += 1;
    }
}
#endif

#ifdef PT_1
// Need to produce this for BOTH input and output
// TODO could make an unordered version of this? i.e. running parallely and adding as completed
template<class SM>
void ord_col_tiling(std::vector<typename SM::itype> &col_breakpoints,
                    std::vector<SM *> &res,
                    typename SM::itype current_position) {
    // Get types
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    // Matrix tiled
    SM *src = res.at(current_position);

    std::vector<SM *> temp_res;

    // Get stats from the original matrox
    iT src_ncols = src->ncols();
    iT src_nrows = src->nrows();

    nT *src_offset_ptr = src->offset_ptr();
    iT *src_ids_ptr = src->ids_ptr();
    vT *src_vals_ptr = src->vals_ptr();

    // TODO might be able to parallelize this section if you try to find the values per row?
    //  But would still have an issue if col seg is empty. => Do DCSC?
    for (iT nth_tile = 0; nth_tile < col_breakpoints.size() - 1; nth_tile++) {
        SM *new_adj = new SM();
        temp_res.push_back(new_adj);
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (iT nth_tile = 0; nth_tile < col_breakpoints.size() - 1; nth_tile++) {
        iT j_start = col_breakpoints.at(nth_tile);
        iT j_end = col_breakpoints.at(nth_tile + 1);

        SM *new_adj = temp_res.at(nth_tile);

        nT new_nvals = 0;
        std::vector<nT> new_offset_ptr_vec;
        std::vector<iT> new_ids_ptr_vec;
        std::vector<vT> new_vals_vec;

#ifdef CCMP
        iT new_nrows = 0;
        std::vector<iT> new_rows_vec;
#endif

        new_offset_ptr_vec.push_back(0);
        for (iT i_i = 0; i_i < src_nrows; i_i += 1) {
            nT first_node_edge = src_offset_ptr[i_i];
            nT last_node_edge = src_offset_ptr[i_i + 1];

#ifdef CCMP
            bool found_nnz = false;
#endif
            for (nT e = first_node_edge; e < last_node_edge; e++) {
                iT u = src_ids_ptr[e];
                if (u >= j_start && u < j_end) {
                    vT val = src_vals_ptr[e];
#ifdef CCMP
                    found_nnz = true;
#endif
                    new_ids_ptr_vec.push_back(u);
                    new_vals_vec.push_back(val);

                    new_nvals += 1;
                } else if (u >= j_end) {
                    break;
                }
            }
#ifdef CCMP
            if (found_nnz) {
                new_offset_ptr_vec.push_back(new_nvals);
                new_rows_vec.push_back(i_i);
                new_nrows += 1;
            }
#else
            new_offset_ptr_vec.push_back(new_nvals);
#endif
        }
#ifdef A_ALLOC
#ifdef CCMP
        nT *new_offset_ptr = (nT *) aligned_alloc(64, (new_nrows + 1) * sizeof(nT));
#else
        nT *new_offset_ptr = (nT *) aligned_alloc(64, (src_nrows + 1) * sizeof(nT));
#endif
#else
#ifdef CCMP
        nT *new_offset_ptr = (nT *) malloc((val_rows + 1) * sizeof(nT));
#else
        nT *new_offset_ptr = (nT *) malloc((src_nrows + 1) * sizeof(nT));
#endif
        nT *new_offset_ptr = (nT *) malloc((src_nrows + 1) * sizeof(nT));
#endif
        std::copy(new_offset_ptr_vec.begin(), new_offset_ptr_vec.end(), new_offset_ptr);


#ifdef A_ALLOC
        iT *new_ids_ptr = (iT *) aligned_alloc(64, (new_nvals) * sizeof(iT));
#else
        iT *new_ids_ptr = (iT *) malloc((new_nvals) * sizeof(iT));
#endif
        std::copy(new_ids_ptr_vec.begin(), new_ids_ptr_vec.end(), new_ids_ptr);

#ifdef A_ALLOC
        vT *new_vals = (vT *) aligned_alloc(64, (new_nvals) * sizeof(vT));
#else
        vT *new_vals = (vT *) malloc((new_nvals) * sizeof(vT));
#endif
        std::copy(new_vals_vec.begin(), new_vals_vec.end(), new_vals);

#ifdef CCMP
#ifdef A_ALLOC
        iT *new_rows = (iT *) aligned_alloc(64, (new_nrows) * sizeof(iT));
#else
        iT *new_rows = (iT *) malloc((val_rows) * sizeof(iT));
#endif
        std::copy(new_rows_vec.begin(), new_rows_vec.end(), new_rows);
        new_adj->import_dcsr(new_nrows,
                             j_end - j_start,
                             new_nvals,
                             new_ids_ptr,
                             new_vals,
                             new_offset_ptr,
                             new_rows);

        new_rows_vec.clear();
        new_rows_vec.shrink_to_fit();
#else
        new_adj->import_csr(src_nrows,
                            j_end - j_start,
                            new_nvals,
                            new_ids_ptr,
                            new_vals,
                            new_offset_ptr);
#endif
        new_offset_ptr_vec.clear();
        new_offset_ptr_vec.shrink_to_fit();
        new_ids_ptr_vec.clear();
        new_ids_ptr_vec.shrink_to_fit();
        new_vals_vec.clear();
        new_vals_vec.shrink_to_fit();
    }

    auto insert_position = res.begin() + current_position;
    res.erase(insert_position);

    for (auto tiled_SM: temp_res) {
        insert_position = res.insert(insert_position, tiled_SM);
        insert_position += 1;
    }
}
#endif

//#ifdef PT_2
//template<class SM>
//void ord_col_tiling(std::vector<typename SM::itype> &col_breakpoints,
//                    std::vector<SM *> &res,
//                    typename SM::itype current_position) {
//    // Get types
//    typedef typename SM::itype iT;
//    typedef typename SM::ntype nT;
//    typedef typename SM::vtype vT;
//
//    // Matrix tiled
//    SM *src = res.at(current_position);
//
//    std::vector<SM *> temp_res;
//
//    // Get stats from the original matrox
//    iT src_ncols = src->ncols();
//    iT src_nrows = src->nrows();
//
//    nT *src_offset_ptr = src->offset_ptr();
//    iT *src_ids_ptr = src->ids_ptr();
//    vT *src_vals_ptr = src->vals_ptr();
//
//    for (iT nth_tile = 0; nth_tile < col_breakpoints.size() - 1; nth_tile++) {
//        SM *new_adj = new SM();
//        temp_res.push_back(new_adj);
//    }
//
//#ifdef DBG_PT2
//    double time_initial_alloc = 0;
//    double time_parallel_search = 0;
//    double time_serial_offset = 0;
//    double time_copy_offset = 0;
//    double time_populate_data = 0;
//    double time_clean = 0;
//    double start_time, start_time_total, time_total;
//    start_time_total = get_time();
////    nT total_es = 0;
//#endif
//
//    auto start_offset_ptr = (nT *) aligned_alloc(64, (src_nrows) * sizeof(nT));
//    std::copy(src_offset_ptr, src_offset_ptr + src_nrows, start_offset_ptr);
//
//    auto row_counter_vec = (nT *) aligned_alloc(64, (src_nrows) * sizeof(nT));
//    auto row_dcsr_vec = (bool *) aligned_alloc(64, (src_nrows) * sizeof(bool));
//
//    for (iT nth_tile = 0; nth_tile < col_breakpoints.size() - 1; nth_tile++) {
//        iT j_start = col_breakpoints.at(nth_tile);
//        iT j_end = col_breakpoints.at(nth_tile + 1);
//
//        SM *new_adj = temp_res.at(nth_tile);
//
//#ifdef DBG_PT2
//        start_time = get_time();
////        std::cout << "Finish init" << std::endl;
//#endif
////#pragma omp parallel for schedule(dynamic, 4) default(none) shared(src_nrows, src_offset_ptr, src_ids_ptr, src_vals_ptr, row_ids_ptr_vec, row_vals_vec, row_counter_vec, row_dcsr_vec, j_end, j_start)
//#pragma omp parallel for schedule(dynamic, 4)
//        for (iT i_i = 0; i_i < src_nrows; i_i += 1) {
//            iT row_nvals = 0;
//
//            nT first_node_edge = start_offset_ptr[i_i];
//            nT last_node_edge = src_offset_ptr[i_i + 1];
//
//            bool found_nnz = false;
//            for (nT e = first_node_edge; e < last_node_edge; e++) {
//                iT u = src_ids_ptr[e];
//                if (u >= j_start && u < j_end) {
//                    found_nnz = true;
//                    row_nvals += 1;
//                } else if (u >= j_end) {
//                    break;
//                }
//            }
//            if (found_nnz) {
//                row_counter_vec[i_i] = row_nvals;
//                row_dcsr_vec[i_i] = true;
//            }else {
//                row_dcsr_vec[i_i] = false;
//            }
//        }
//#ifdef DBG_PT2
//        time_parallel_search += (get_time() - start_time);
////        std::cout << "Finish parallel search" << std::endl;
//#endif
//        nT new_nvals = 0;
//        iT new_nrows = 0;
//
//        std::vector<nT> new_offset_ptr_vec;
//        std::vector<iT> new_rows_vec;
//
//        // TODO figure out a way to improve the speed here (Blelloch?)
//        new_offset_ptr_vec.push_back(0);
//#ifdef DBG_PT2
//        start_time = get_time();
//#endif
//        for (iT ith_row = 0; ith_row < src_nrows; ith_row++) {
//            if (row_dcsr_vec[ith_row]) {
//                new_nrows += 1;
//                new_nvals += row_counter_vec[ith_row];
//                new_offset_ptr_vec.push_back(new_nvals);
//                new_rows_vec.push_back(ith_row);
//            }
//        }
//#ifdef DBG_PT2
////        total_es += new_nvals;
////        std::cout << "e:" << new_nvals << std::endl;
//        time_serial_offset += (get_time() - start_time);
////        std::cout << "Finish serial offset" << std::endl;
//#endif
//
//#ifdef DBG_PT2
//        start_time = get_time();
//#endif
//        nT *new_offset_ptr = (nT *) aligned_alloc(64, (new_nrows + 1) * sizeof(nT));
//        std::copy(new_offset_ptr_vec.begin(), new_offset_ptr_vec.end(), new_offset_ptr);
//        iT *new_rows = (iT *) aligned_alloc(64, (new_nrows) * sizeof(iT));
//        std::copy(new_rows_vec.begin(), new_rows_vec.end(), new_rows);
//
//        iT *new_ids_ptr = (iT *) aligned_alloc(64, (new_nvals) * sizeof(iT));
//        vT *new_vals = (vT *) aligned_alloc(64, (new_nvals) * sizeof(vT));
//#ifdef DBG_PT2
//        time_copy_offset += (get_time() - start_time);
////        std::cout << "Finish copy offset" << std::endl;
//#endif
//
//#ifdef DBG_PT2
//        start_time = get_time();
//#endif
//        // Don't need to do all rows. Just do the ones with values.
//#pragma omp parallel for schedule(dynamic, 1)
//        for (iT i_r = 0; i_r < new_nrows; i_r++) {
//            iT node_i = new_rows[i_r];
//
//            nT first_node_edge = new_offset_ptr[i_r];
//            nT last_node_edge = new_offset_ptr[i_r + 1];
//            nT nr_vals = last_node_edge - first_node_edge;
//
//            nT src_first_edge = start_offset_ptr[node_i];
//
//            for (nT e = 0; e < nr_vals; e++) {
//                nT dst_node = first_node_edge + e;
//                nT src_node = src_first_edge + e;
//                new_ids_ptr[dst_node] = src_ids_ptr[src_node];
//                new_vals[dst_node] = src_vals_ptr[src_node];
//            }
//            start_offset_ptr[node_i] = start_offset_ptr[node_i] + nr_vals;
//        }
//#ifdef DBG_PT2
//        time_populate_data += (get_time() - start_time);
////        std::cout << "Finish data populate" << std::endl;
//#endif
//
//        new_adj->import_dcsr(new_nrows,
//                             j_end - j_start,
//                             new_nvals,
//                             new_ids_ptr,
//                             new_vals,
//                             new_offset_ptr,
//                             new_rows);
//
//#ifdef DBG_PT2
//        start_time = get_time();
//#endif
//        new_rows_vec.clear();
//        new_rows_vec.shrink_to_fit();
//        new_offset_ptr_vec.clear();
//        new_offset_ptr_vec.shrink_to_fit();
//#ifdef DBG_PT2
//        time_clean += (get_time() - start_time);
////        std::cout << "Finish clean" << std::endl;
//#endif
//    }
//
//    auto insert_position = res.begin() + current_position;
//    res.erase(insert_position);
//    free(start_offset_ptr);
//    free(row_counter_vec);
//    free(row_dcsr_vec);
//
//    for (auto tiled_SM: temp_res) {
//        insert_position = res.insert(insert_position, tiled_SM);
//        insert_position += 1;
//    }
//
//#ifdef DBG_PT2
//    time_total = (get_time() - start_time_total);
//    std::cout << "Debug times of PT2" << std::endl;
//    std::cout << "Parallel search: " << time_parallel_search << " , percen: " << time_parallel_search * 100 / time_total
//              << std::endl;
//    std::cout << "Serial offset: " << time_serial_offset << " , percen: " << time_serial_offset * 100 / time_total
//              << std::endl;
//    std::cout << "Copy offset: " << time_copy_offset << " , percen: " << time_copy_offset * 100 / time_total
//              << std::endl;
//    std::cout << "Populate data: " << time_populate_data << " , percen: " << time_populate_data * 100 / time_total
//              << std::endl;
//    std::cout << "Clean: " << time_clean << " , percen: " << time_clean * 100 / time_total
//              << std::endl;
////    std::cout << "es: " << total_es << std::endl;
//#endif
//}
//#endif

template<class DM>
void slice_tiling(typename DM::itype slice_size,
                  DM *src,
                  std::vector<DM *> &res) {
    // Get types
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    // Full size of the input dense matrix
    diT src_ncols = src->ncols();
    diT src_nrows = src->nrows();
    dvT *src_vals_ptr = src->vals_ptr();

    // Create a slice of the input dense matrix
    for (diT k_0 = 0; k_0 < src_ncols; k_0 += slice_size) {
        DM *new_dense = new DM();
        diT k_num = std::min(slice_size, src_ncols - k_0);
        auto local_vals = (dvT *) aligned_alloc(64, (src_nrows * k_num) * sizeof(dvT));

#pragma omp parallel for schedule(static, 4)
        for (diT i_i = 0; i_i < src_nrows; i_i += 1) {
            for (diT k_i = k_0; k_i < k_0 + k_num; k_i += 1) {
                local_vals[(dnT)(i_i * k_num + (k_i - k_0))] = src_vals_ptr[(dnT)(i_i * src_ncols + k_i)];
            }
        }
        new_dense->import_mtx(src_nrows, k_num, src_nrows * k_num, local_vals);
        res.push_back(new_dense);
    }
}

#ifdef PT_2

template<class SM>
void ord_col_tiling(std::vector<typename SM::itype> &col_breakpoints,
                    std::vector<SM *> &res,
                    typename SM::itype current_position) {
    // Get types
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    // Matrix tiled
    SM *src = res.at(current_position);

    std::vector<SM *> temp_res;

    // Get stats from the original matrox
    iT src_ncols = src->ncols();
    iT src_nrows = src->nrows();

    nT *src_offset_ptr = src->offset_ptr();
    iT *src_ids_ptr = src->ids_ptr();
    vT *src_vals_ptr = src->vals_ptr();

    for (iT nth_tile = 0; nth_tile < col_breakpoints.size() - 1; nth_tile++) {
        SM *new_adj = new SM();
        temp_res.push_back(new_adj);
    }

#ifdef DBG_PT2
    double time_initial_alloc = 0;
    double time_parallel_search = 0;
    double time_serial_offset = 0;
    double time_copy_offset = 0;
    double time_populate_data = 0;
    double time_clean = 0;
    double start_time, start_time_total, time_total;
    start_time_total = get_time();
//    nT total_es = 0;
#endif

    auto start_offset_ptr = (nT *) aligned_alloc(64, (src_nrows) * sizeof(nT));
    std::copy(src_offset_ptr, src_offset_ptr + src_nrows, start_offset_ptr);

    auto row_counter_vec = (nT *) aligned_alloc(64, (src_nrows) * sizeof(nT));
    auto row_dcsr_vec = (bool *) aligned_alloc(64, (src_nrows) * sizeof(bool));

    for (iT nth_tile = 0; nth_tile < col_breakpoints.size() - 1; nth_tile++) {
        iT j_start = col_breakpoints.at(nth_tile);
        iT j_end = col_breakpoints.at(nth_tile + 1);

        SM *new_adj = temp_res.at(nth_tile);

#ifdef DBG_PT2
        start_time = get_time();
//        std::cout << "Finish init" << std::endl;
#endif
//#pragma omp parallel for schedule(dynamic, 4) default(none) shared(src_nrows, src_offset_ptr, src_ids_ptr, src_vals_ptr, row_ids_ptr_vec, row_vals_vec, row_counter_vec, row_dcsr_vec, j_end, j_start)
#pragma omp parallel for schedule(dynamic, 4)
        for (iT i_i = 0; i_i < src_nrows; i_i += 1) {
            iT row_nvals = 0;

            nT first_node_edge = start_offset_ptr[i_i];
            nT last_node_edge = src_offset_ptr[i_i + 1];

            bool found_nnz = false;
            for (nT e = first_node_edge; e < last_node_edge; e++) {
                iT u = src_ids_ptr[e];
                if (u >= j_start && u < j_end) {
                    found_nnz = true;
                    row_nvals += 1;
                } else if (u >= j_end) {
                    break;
                }
            }
            if (found_nnz) {
                row_counter_vec[i_i] = row_nvals;
                row_dcsr_vec[i_i] = true;
            } else {
                row_dcsr_vec[i_i] = false;
            }
        }
#ifdef DBG_PT2
        time_parallel_search += (get_time() - start_time);
//        std::cout << "Finish parallel search" << std::endl;
#endif
        nT new_nvals = 0;
#ifdef DBG_PT2
        start_time = get_time();
#endif
        iT new_nrows = __gnu_parallel::count(row_dcsr_vec, row_dcsr_vec + src_nrows, true);
        nT *new_offset_ptr = (nT *) aligned_alloc(64, (new_nrows + 1) * sizeof(nT));
        iT *new_rows = (iT *) aligned_alloc(64, (new_nrows) * sizeof(iT));
        new_offset_ptr[0] = 0;
        iT r_i = 0;
        for (iT ith_row = 0; ith_row < src_nrows; ith_row++) {
            if (row_dcsr_vec[ith_row]) {
                new_nvals += row_counter_vec[ith_row];
                new_rows[r_i++] = ith_row;
                new_offset_ptr[r_i] = new_nvals;
            }
        }
#ifdef DBG_PT2
        //        total_es += new_nvals;
        //        std::cout << "e:" << new_nvals << std::endl;
                time_serial_offset += (get_time() - start_time);
        //        std::cout << "Finish serial offset" << std::endl;
#endif

#ifdef DBG_PT2
        start_time = get_time();
#endif
        iT *new_ids_ptr = (iT *) aligned_alloc(64, (new_nvals) * sizeof(iT));
        vT *new_vals = (vT *) aligned_alloc(64, (new_nvals) * sizeof(vT));
#ifdef DBG_PT2
        time_copy_offset += (get_time() - start_time);
#endif

#ifdef DBG_PT2
        start_time = get_time();
#endif
        // Don't need to do all rows. Just do the ones with values.
#pragma omp parallel for schedule(dynamic, 4)
        for (iT i_r = 0; i_r < new_nrows; i_r++) {
            iT node_i = new_rows[i_r];

            nT first_node_edge = new_offset_ptr[i_r];
            nT last_node_edge = new_offset_ptr[i_r + 1];
            nT nr_vals = last_node_edge - first_node_edge;

            nT src_first_edge = start_offset_ptr[node_i];

            for (nT e = 0; e < nr_vals; e++) {
                nT dst_node = first_node_edge + e;
                nT src_node = src_first_edge + e;
                new_ids_ptr[dst_node] = src_ids_ptr[src_node];
                new_vals[dst_node] = src_vals_ptr[src_node];
            }
            start_offset_ptr[node_i] = start_offset_ptr[node_i] + nr_vals;
        }
#ifdef DBG_PT2
        time_populate_data += (get_time() - start_time);
//        std::cout << "Finish data populate" << std::endl;
#endif

        new_adj->import_dcsr(new_nrows,
                             j_end - j_start,
                             new_nvals,
                             new_ids_ptr,
                             new_vals,
                             new_offset_ptr,
                             new_rows);

#ifdef DBG_PT2
        start_time = get_time();
#endif
#ifdef DBG_PT2
        time_clean += (get_time() - start_time);
//        std::cout << "Finish clean" << std::endl;
#endif
    }

    auto insert_position = res.begin() + current_position;
    res.erase(insert_position);
    free(start_offset_ptr);
    free(row_counter_vec);
    free(row_dcsr_vec);

    for (auto tiled_SM: temp_res) {
        insert_position = res.insert(insert_position, tiled_SM);
        insert_position += 1;
    }

#ifdef DBG_PT2
    time_total = (get_time() - start_time_total);
    std::cout << "Debug times of PT2" << std::endl;
    std::cout << "Parallel search: " << time_parallel_search << " , percen: " << time_parallel_search * 100 / time_total
              << std::endl;
    std::cout << "Serial offset: " << time_serial_offset << " , percen: " << time_serial_offset * 100 / time_total
              << std::endl;
    std::cout << "Copy offset: " << time_copy_offset << " , percen: " << time_copy_offset * 100 / time_total
              << std::endl;
    std::cout << "Populate data: " << time_populate_data << " , percen: " << time_populate_data * 100 / time_total
              << std::endl;
    std::cout << "Clean: " << time_clean << " , percen: " << time_clean * 100 / time_total
              << std::endl;
//    std::cout << "es: " << total_es << std::endl;
#endif
}

#endif


/// FOR SPADE!!
template<class SM>
void ord_col_tiling_SPADE(std::vector<typename SM::itype> &col_breakpoints,
                          std::vector<SM *> &res,
                          typename SM::itype current_position) {
    // Get types
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    // Matrix tiled
    SM *src = res.at(current_position);

    std::vector<SM *> temp_res;

    // Get stats from the original matrox
    iT src_ncols = src->ncols();
    iT src_nrows = src->nrows();

    nT *src_offset_ptr = src->offset_ptr();
    iT *src_ids_ptr = src->ids_ptr();
    vT *src_vals_ptr = src->vals_ptr();

    // TODO might be able to parallelize this section if you try to find the values per row?
    //  But would still have an issue if col seg is empty. => Do DCSC?
    for (iT nth_tile = 0; nth_tile < col_breakpoints.size() - 1; nth_tile++) {
        SM *new_adj = new SM();
        temp_res.push_back(new_adj);
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (iT nth_tile = 0; nth_tile < col_breakpoints.size() - 1; nth_tile++) {
        iT j_start = col_breakpoints.at(nth_tile);
        iT j_end = col_breakpoints.at(nth_tile + 1);

        SM *new_adj = temp_res.at(nth_tile);

        nT new_nvals = 0;
        std::vector<nT> new_offset_ptr_vec;
        std::vector<iT> new_ids_ptr_vec;
        std::vector<vT> new_vals_vec;

        new_offset_ptr_vec.push_back(0);
        for (iT i_i = 0; i_i < src_nrows; i_i += 1) {
            nT first_node_edge = src_offset_ptr[i_i];
            nT last_node_edge = src_offset_ptr[i_i + 1];

            for (nT e = first_node_edge; e < last_node_edge; e++) {
                iT u = src_ids_ptr[e];
                if (u >= j_start && u < j_end) {
                    vT val = src_vals_ptr[e];
                    new_ids_ptr_vec.push_back(u);
                    new_vals_vec.push_back(val);

                    new_nvals += 1;
                } else if (u >= j_end) {
                    break;
                }
            }

            new_offset_ptr_vec.push_back(new_nvals);
        }
#ifdef A_ALLOC
        nT *new_offset_ptr = (nT *) aligned_alloc(64, (src_nrows + 1) * sizeof(nT));
#else
        nT *new_offset_ptr = (nT *) malloc((src_nrows + 1) * sizeof(nT));
#endif
        std::copy(new_offset_ptr_vec.begin(), new_offset_ptr_vec.end(), new_offset_ptr);


#ifdef A_ALLOC
        iT *new_ids_ptr = (iT *) aligned_alloc(64, (new_nvals) * sizeof(iT));
#else
        iT *new_ids_ptr = (iT *) malloc((new_nvals) * sizeof(iT));
#endif
        std::copy(new_ids_ptr_vec.begin(), new_ids_ptr_vec.end(), new_ids_ptr);

#ifdef A_ALLOC
        vT *new_vals = (vT *) aligned_alloc(64, (new_nvals) * sizeof(vT));
#else
        vT *new_vals = (vT *) malloc((new_nvals) * sizeof(vT));
#endif
        std::copy(new_vals_vec.begin(), new_vals_vec.end(), new_vals);

        new_adj->import_csr(src_nrows,
                            j_end - j_start,
                            new_nvals,
                            new_ids_ptr,
                            new_vals,
                            new_offset_ptr);
        new_offset_ptr_vec.clear();
        new_offset_ptr_vec.shrink_to_fit();
        new_ids_ptr_vec.clear();
        new_ids_ptr_vec.shrink_to_fit();
        new_vals_vec.clear();
        new_vals_vec.shrink_to_fit();
    }

    auto insert_position = res.begin() + current_position;
    res.erase(insert_position);

    for (auto tiled_SM: temp_res) {
        insert_position = res.insert(insert_position, tiled_SM);
        insert_position += 1;
    }
}

// Need to produce this for BOTH input and output
// TODO could make an unordered version of this? i.e. running parallely and adding as completed
template<class SM>
void ord_col_tiling_AU(std::vector<typename SM::itype> &col_breakpoints,
                       std::vector<SM *> &res,
                       typename SM::itype current_position) {
    // Get types
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    // Matrix tiled
    SM *src = res.at(current_position);

    std::vector<SM *> temp_res;

    // Get stats from the original matrox
    iT src_ncols = src->ncols();
    iT src_nrows = src->nrows();

    nT *src_offset_ptr = src->offset_ptr();
    iT *src_ids_ptr = src->ids_ptr();
    vT *src_vals_ptr = src->vals_ptr();

#ifdef NEXT_GEMM
    bool *next_work = (bool *) malloc((src_nrows) * sizeof(bool));
    for (iT i_i = 0; i_i < src_nrows; i_i += 1) {
        next_work[i_i] = false;
    }
#endif

    for (iT nth_tile = 0; nth_tile < col_breakpoints.size() - 1; nth_tile++) {
//        std::cerr << nth_tile << std::endl;
        iT j_start = col_breakpoints.at(nth_tile);
        iT j_end = col_breakpoints.at(nth_tile + 1);

        SM *new_adj = new SM();

        nT new_nvals = 0;
        std::vector<nT> new_offset_ptr_vec;
        std::vector<iT> new_ids_ptr_vec;
        std::vector<vT> new_vals_vec;

        bool *new_work = (bool *) malloc((src_nrows) * sizeof(bool));
        for (iT i_i = 0; i_i < src_nrows; i_i += 1) {
            new_work[i_i] = false;
        }

        new_offset_ptr_vec.push_back(0);
        for (iT i_i = 0; i_i < src_nrows; i_i += 1) {
            nT first_node_edge = src_offset_ptr[i_i];
            nT last_node_edge = src_offset_ptr[i_i + 1];
            for (nT e = first_node_edge; e < last_node_edge; e++) {
                iT u = src_ids_ptr[e];
                if (u >= j_start && u < j_end) {
#ifdef REV_JJ
#ifndef LAST_GEMM
                    if (e == first_node_edge){
                        new_work[i_i] = true;
                    }
#endif
#endif
                    vT val = src_vals_ptr[e];

                    new_ids_ptr_vec.push_back(u);
                    new_vals_vec.push_back(val);

                    new_nvals += 1;

#ifndef REV_JJ
#ifdef NEXT_GEMM
                    if (e == last_node_edge - 1) {
                        if (nth_tile == col_breakpoints.size() - 2){
                            new_work[i_i] = true;
                        } else {
                            next_work[i_i] = true;
                        }
                    }
#else
                    if (e == last_node_edge - 1) {
                        new_work[i_i] = true;
                    }
#endif
#endif
                }
            }
#ifdef NEXT_GEMM
            if (next_work[i_i]) {
                new_work[i_i] = true;
                next_work[i_i] = false;
            }
#endif
#ifdef LAST_GEMM
            if (nth_tile == 0) {
                new_work[i_i] = true;
            }
#endif
            new_offset_ptr_vec.push_back(new_nvals);
        }
        nT *new_offset_ptr = (nT *) malloc((src_nrows + 1) * sizeof(nT));
        std::copy(new_offset_ptr_vec.begin(), new_offset_ptr_vec.end(), new_offset_ptr);
        iT *new_ids_ptr = (iT *) malloc((new_nvals) * sizeof(iT));
        std::copy(new_ids_ptr_vec.begin(), new_ids_ptr_vec.end(), new_ids_ptr);
        vT *new_vals = (vT *) malloc((new_nvals) * sizeof(vT));
        std::copy(new_vals_vec.begin(), new_vals_vec.end(), new_vals);

        new_adj->import_csr(src_nrows,
                            j_end - j_start,
                            new_nvals,
                            new_ids_ptr,
                            new_vals,
                            new_offset_ptr);
        new_adj->import_work_mask(new_work);

        temp_res.push_back(new_adj);

        new_offset_ptr_vec.clear();
        new_ids_ptr_vec.clear();
        new_vals_vec.clear();
    }
    auto insert_position = res.begin() + current_position;
    res.erase(insert_position);

    for (auto tiled_SM: temp_res) {
        insert_position = res.insert(insert_position, tiled_SM);
        insert_position += 1;
    }
}


// TODO try to add this to the other tiling? (These create a very high replication of code)
// padded column tiling
template<class SM>
void ord_col_tiling_static_padding(std::vector<typename SM::itype> &col_breakpoints,
                                   std::vector<SM *> &res,
                                   typename SM::itype current_position,
                                   typename SM::itype rowtile_size,
                                   typename SM::itype multiple_to_pad_to) {
    // Get types
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    // Matrix tiled
    SM *src = res.at(current_position);

    std::vector<SM *> temp_res;

    // Get stats from the original matrox
    iT src_ncols = src->ncols();
    iT src_nrows = src->nrows();

    nT *src_offset_ptr = src->offset_ptr();
    iT *src_ids_ptr = src->ids_ptr();
    vT *src_vals_ptr = src->vals_ptr();

    // TODO you can parallelize / tile this as well

    for (iT nth_tile = 0; nth_tile < col_breakpoints.size() - 1; nth_tile++) {
//        std::cerr << nth_tile << std::endl;
        iT j_start = col_breakpoints.at(nth_tile);
        iT j_end = col_breakpoints.at(nth_tile + 1);

        SM *new_adj = new SM();

        nT new_nvals = 0;
        std::vector<nT> new_offset_ptr_vec;
        std::vector<iT> new_ids_ptr_vec;
        std::vector<vT> new_vals_vec;

        new_offset_ptr_vec.push_back(0);
        for (iT i_i = 0; i_i < src_nrows; i_i += 1) {
            nT first_node_edge = src_offset_ptr[i_i];
            nT last_node_edge = src_offset_ptr[i_i + 1];

            iT last_id;
            for (nT e = first_node_edge; e < last_node_edge; e++) {
                iT u = src_ids_ptr[e];
                if (u >= j_start && u < j_end) {
                    vT val = src_vals_ptr[e];

                    new_ids_ptr_vec.push_back(u);
                    last_id = u;

                    new_vals_vec.push_back(val);

                    new_nvals += 1;
                }
            }
            if ((i_i + 1) % rowtile_size == 0 || (i_i + 1) == src_nrows) {
                iT mod_div = new_nvals % multiple_to_pad_to;

                if (mod_div != 0) {
                    for (int pad_i = 0; pad_i < multiple_to_pad_to - mod_div; pad_i++) {
                        new_ids_ptr_vec.push_back(last_id);
                        new_vals_vec.push_back(0);
                        new_nvals += 1;
                    }
                }
            }
            new_offset_ptr_vec.push_back(new_nvals);
        }
        nT *new_offset_ptr = (nT *) malloc((src_nrows + 1) * sizeof(nT));
        std::copy(new_offset_ptr_vec.begin(), new_offset_ptr_vec.end(), new_offset_ptr);
        iT *new_ids_ptr = (iT *) malloc((new_nvals) * sizeof(iT));
        std::copy(new_ids_ptr_vec.begin(), new_ids_ptr_vec.end(), new_ids_ptr);
        vT *new_vals = (vT *) malloc((new_nvals) * sizeof(vT));
        std::copy(new_vals_vec.begin(), new_vals_vec.end(), new_vals);

        new_adj->import_csr(src_nrows,
                            j_end - j_start,
                            new_nvals,
                            new_ids_ptr,
                            new_vals,
                            new_offset_ptr);

        temp_res.push_back(new_adj);
        new_offset_ptr_vec.clear();
        new_ids_ptr_vec.clear();
        new_vals_vec.clear();
    }
    auto insert_position = res.begin() + current_position;
    res.erase(insert_position);

    for (auto tiled_SM: temp_res) {
        insert_position = res.insert(insert_position, tiled_SM);
        insert_position += 1;
    }
}

// Functions that specify the partitions
// Can be ordered or not (if ordered, then result is the offset to split by, else, is the columns in one parition)
/**
 * Static split of columns - Ordered
 * @tparam itype - Type of the node id
 */
template<class SM>
std::vector<typename SM::itype> static_ord_col_breakpoints(SM *mtx,
                                                           typename SM::itype cols_per_partition) {
    // Get types
    typedef typename SM::itype iT;

    std::vector<iT> res;
    res.push_back(0);

    for (iT i = 0; i < mtx->ncols(); i += cols_per_partition) {
        iT part_end = std::min(mtx->ncols(), i + cols_per_partition);
        res.push_back(part_end);
    }
    return res;
}

/**
 * Static split of rows
 * @tparam itype - Type of the node id
 */
template<class SM>
std::vector<typename SM::itype> static_ord_row_breakpoints(SM *mtx,
                                                           typename SM::itype rows_per_partition) {
    // Get types
    typedef typename SM::itype iT;

    std::vector<iT> res;
    res.push_back(0);

    for (iT i = 0; i < mtx->nrows(); i += rows_per_partition) {
        iT part_end = std::min(mtx->nrows(), i + rows_per_partition);
        res.push_back(part_end);
    }
    return res;
}

template<class SM, class DM>
void static_ord_row_tile_info(std::vector<SM *> tiled_adj,
                              typename SM::itype rows_per_partition,
                              std::vector<std::vector<GNOpTile<SM, DM>>> &tile_infos) {
    // Get types
    typedef typename SM::itype iT;

    for (iT t_j = 0; t_j < tiled_adj.size(); t_j++) {
        auto adj_nrows = tiled_adj.at(t_j)->nrows();
        auto adj_offsets = tiled_adj.at(t_j)->offset_ptr();

        std::vector<GNOpTile<SM, DM>> col_seg_tile_infos;
        for (iT t_i = 0; t_i < adj_nrows; t_i += rows_per_partition) {
            GNOpTile<SM, DM> tile_info;

            tile_info.srows_start = t_i;
            tile_info.srows_end = std::min(adj_nrows, t_i + rows_per_partition);

            if (adj_offsets[tile_info.srows_start] != adj_offsets[tile_info.srows_end + 1]) {
                col_seg_tile_infos.push_back(tile_info);
            }
        }
        tile_infos.push_back(col_seg_tile_infos);
    }
}

template<class SM, class DM>
void nnz_ord_row_tile_info(std::vector<SM *> tiled_adj,
                           typename SM::itype break_into,
                           std::vector<std::vector<GNOpTile<SM, DM>>> &tile_infos) {
    // Get types
    typedef typename SM::itype iT;
    typedef typename SM::itype nT;

    for (iT t_j = 0; t_j < tiled_adj.size(); t_j++) {
        auto adj_nrows = tiled_adj.at(t_j)->nrows();
        auto adj_offsets = tiled_adj.at(t_j)->offset_ptr();
        auto adj_nnzs = tiled_adj.at(t_j)->nvals();

        auto nnzs_per_tile = adj_nnzs / break_into;
        if (nnzs_per_tile < 1) {
            nnzs_per_tile = 1;
        }

        nT prev_offset = 0;
        nT prev_count = 0;
        iT prev_row = 0;

//        std::cout << "Endpoint: " << nnzs_per_tile << std::endl;

        std::vector<GNOpTile<SM, DM>> col_seg_tile_infos;
        for (iT t_i = 0; t_i < adj_nrows; t_i++) {
            auto current_offset_start = adj_offsets[t_i];
            auto current_offset_end = adj_offsets[t_i + 1];

            // Account for sections with no work
            if (prev_offset == current_offset_start) {
                prev_row = t_i;
            }

            if (current_offset_end >= (prev_count + nnzs_per_tile)) {
                GNOpTile<SM, DM> tile_info;

                tile_info.srows_start = prev_row;
                tile_info.srows_end = t_i + 1;

//                std::cout << tile_info.srows_start << " " << tile_info.srows_end << " " << current_offset_end << std::endl;

                col_seg_tile_infos.push_back(tile_info);

                prev_row = t_i + 1;
                prev_count += nnzs_per_tile;
                prev_offset = current_offset_end;

            }
        }
        tile_infos.push_back(col_seg_tile_infos);
    }
}

template<class T1, class T2>
bool revComparePair(std::pair<T1, T2> i1, std::pair<T1, T2> i2) {
    return (i1.first > i2.first);
}

template<class SM, class DM>
void sort_nnz_row_tile_info(std::vector<SM *> tiled_adj,
                            std::vector<std::vector<GNOpTile<SM, DM>>> &tile_infos,
                            std::vector<std::vector<GNOpTile<SM, DM>>> &res) {
    // Get types
    typedef typename SM::itype iT;
    typedef typename SM::itype nT;

    for (iT t_j = 0; t_j < tiled_adj.size(); t_j++) {
        auto adj_offsets = tiled_adj.at(t_j)->offset_ptr();
        auto tile_segment = tile_infos.at(t_j);
        auto ntiles = tile_segment.size();

        std::vector<GNOpTile<SM, DM>> new_tile_segment;

        std::pair<nT, iT> temp_pair[ntiles];

        // Storing the respective array
        // elements in pairs.
#pragma omp parallel for
        for (iT i = 0; i < ntiles; i++) {
            auto tile_info = tile_segment.at(i);

            auto current_offset_start = adj_offsets[tile_info.srows_start];
            auto current_offset_end = adj_offsets[tile_info.srows_end];

            temp_pair[i].first = (current_offset_end - current_offset_start);
            temp_pair[i].second = i;
        }

        // Sorting the pair array.
        sort(temp_pair, temp_pair + ntiles, revComparePair<nT, iT>);

        // Modifying original arrays
        for (iT i = 0; i < ntiles; i++) {
//            std::cout << temp_pair[i].second << " " << temp_pair[i].first << std::endl;
            new_tile_segment.push_back(tile_segment.at(temp_pair[i].second));
        }
//        std::cout << "A" << std::endl;
        res.push_back(new_tile_segment);
    }
}


#endif //SPARSE_ACCELERATOR_TILING_H
