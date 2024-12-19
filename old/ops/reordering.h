//
// Created by damitha on 2/20/22.
//
#include <map>
#include <omp.h>
#include <vector>

#ifdef ICC_CMP
// #include "pstl/execution"
// #include "pstl/algorithm"
#endif

#ifndef SPARSE_ACCELERATOR_REORDERING_H
#define SPARSE_ACCELERATOR_REORDERING_H

// template<class SM>
// std::vector<SM *> splitAdj(SM *adj, int sparse_segments, int col_dense) {
//     // Sparse input (Graph)
//     typedef typename SM::itype iT;
//     typedef typename SM::ntype nT;
//     typedef typename SM::vtype vT;
//
//     iT nrows = adj->nrows();
//     iT ncols = adj->ncols();
//     iT sparse_tile_rows = nrows / sparse_segments;
//
//     nT *adj_offset_ptr = adj->offset_ptr();
//     iT *adj_ids_ptr = adj->ids_ptr();
//     vT *adj_vals_ptr = adj->vals_ptr();
//
//
//     std::vector<SM *> res;
//
//     nT d_nvals = 0;
//     std::vector<iT> d_row_ids_vec;
//     std::vector<iT> d_col_ids_vec;
//     std::vector<vT> d_vals_vec;
//
//     nT s_nvals = 0;
//     std::vector<iT> s_row_ids_vec;
//     std::vector<iT> s_col_ids_vec;
//     std::vector<vT> s_vals_vec;
//
//     for (iT i = 0; i < adj->nrows(); i += sparse_tile_rows) {
//         std::map<nT, int> count_dense;
//
//         iT last_row = std::min(i + sparse_tile_rows, adj->nrows());
//         // Count the uses
//         for (iT i_i = i; i_i < last_row; i_i++) {
//             nT first_node_edge = adj_offset_ptr[i_i];
//             nT last_node_edge = adj_offset_ptr[i_i + 1];
//             for (nT e = first_node_edge; e < last_node_edge; e++) {
//                 iT u = adj_ids_ptr[e];
////                std::cout << i_i << " " << u << " " << adj->nvals() <<
///std::endl;
//                if (count_dense.count(u)) {
//                    count_dense[u] += 1;
//                } else {
//                    count_dense[u] = 1;
//                }
//            }
//        }
//
//        // Assign COO value to Dense or sparse segment
//        for (iT i_i = i; i_i < last_row; i_i++) {
//            nT first_node_edge = adj_offset_ptr[i_i];
//            nT last_node_edge = adj_offset_ptr[i_i + 1];
//            for (nT e = first_node_edge; e < last_node_edge; e++) {
//                iT u = adj_ids_ptr[e];
//                vT val = adj_vals_ptr[e];
//                if (count_dense[u] > col_dense) {
//                    d_nvals += 1;
//                    d_row_ids_vec.push_back(i_i);
//                    d_col_ids_vec.push_back(u);
//                    d_vals_vec.push_back(val);
//                } else {
//                    s_nvals += 1;
//                    s_row_ids_vec.push_back(i_i);
//                    s_col_ids_vec.push_back(u);
//                    s_vals_vec.push_back(val);
//                }
//            }
//        }
//    }
//
//    iT *d_row_ids = (iT *) alloca((d_nvals) * sizeof(iT));
//    std::copy(d_row_ids_vec.begin(), d_row_ids_vec.end(), d_row_ids);
//    iT *d_col_ids = (iT *) alloca((d_nvals) * sizeof(iT));
//    std::copy(d_col_ids_vec.begin(), d_col_ids_vec.end(), d_col_ids);
//    vT *d_vals = (vT *) alloca((d_nvals) * sizeof(vT));
//    std::copy(d_vals_vec.begin(), d_vals_vec.end(), d_vals);
//
//    iT *s_row_ids = (iT *) alloca((s_nvals) * sizeof(iT));
//    std::copy(s_row_ids_vec.begin(), s_row_ids_vec.end(), s_row_ids);
//    iT *s_col_ids = (iT *) alloca((s_nvals) * sizeof(iT));
//    std::copy(s_col_ids_vec.begin(), s_col_ids_vec.end(), s_col_ids);
//    vT *s_vals = (vT *) alloca((s_nvals) * sizeof(vT));
//    std::copy(s_vals_vec.begin(), s_vals_vec.end(), s_vals);
//
//    SM *new_dense_adj = new SM;
//    new_dense_adj->build(nrows, ncols, d_nvals, d_row_ids, d_col_ids, d_vals,
//    adj->type());
//
//    SM *new_sparse_adj = new SM;
//    new_sparse_adj->build(nrows, ncols, s_nvals, s_row_ids, s_col_ids, s_vals,
//    adj->type());
//
//    res.push_back(new_dense_adj);
//    res.push_back(new_sparse_adj);
//
//    return res;
//}

template <class SM>
std::vector<typename SM::itype> getASpTDense(SM *adj, int col_dense) {
  // Sparse input (Graph)
  typedef typename SM::itype iT;
  typedef typename SM::ntype nT;
  typedef typename SM::vtype vT;

  iT nrows = adj->nrows();
  iT ncols = adj->ncols();

  nT *adj_offset_ptr = adj->offset_ptr();
  iT *adj_ids_ptr = adj->ids_ptr();

  std::vector<iT> count_uses;
  for (iT col = 0; col < ncols; col++) {
    count_uses.push_back(0);
  }
  std::vector<iT> dense_cols;

  iT last_row = adj->nrows();
  // Count the uses
  for (iT i_i = 0; i_i < last_row; i_i++) {
    nT first_node_edge = adj_offset_ptr[i_i];
    nT last_node_edge = adj_offset_ptr[i_i + 1];
    for (nT e = first_node_edge; e < last_node_edge; e++) {
      iT u = adj_ids_ptr[e];
      count_uses.at(0) += 1;
    }
  }

  // Would give the result in sorted order
  for (iT col = 0; col < ncols; col++) {
    if (count_uses.at(0) > col_dense) {
      dense_cols.push_back(col);
    }
  }

  return dense_cols;
}

// Why can't the permutation be a function??
template <class SM> void rowReorder(SM *adj, typename SM::itype *perm) {
  // Sparse input (Graph)
  typedef typename SM::itype iT;
  typedef typename SM::ntype nT;
  typedef typename SM::vtype vT;

  iT nrows = adj->nrows();
  iT ncols = adj->ncols();
  nT nvals = adj->nvals();

  nT *adj_offset_ptr = adj->offset_ptr();
  iT *adj_ids_ptr = adj->ids_ptr();
  vT *adj_vals_ptr = adj->vals_ptr();

  // TODO Would it be more efficient to have this in alloca??
  nT *new_offset_ptr = (nT *)malloc((nrows + 1) * sizeof(nT));
  iT *new_ids_ptr = (iT *)malloc((nvals) * sizeof(iT));
  vT *new_vals_ptr = (vT *)malloc((nvals) * sizeof(vT));

  new_offset_ptr[0] = 0;

#pragma omp parallel
  for (iT i = 0; i < adj->nrows(); i += 1) {
    iT new_row = perm[i];
    nT new_pre_offset = new_offset_ptr[i];

    nT pre_offset = adj_offset_ptr[new_row];
    nT post_offset = adj_offset_ptr[new_row + 1];

    new_offset_ptr[i + 1] = new_pre_offset + (post_offset - pre_offset);

    for (nT e = pre_offset; e < post_offset; e++) {
      new_ids_ptr[new_pre_offset] = adj_ids_ptr[e];
      new_vals_ptr[new_pre_offset] = adj_vals_ptr[e];
      new_pre_offset += 1;
    }
  }

  // TODO change this to matrix import (from csr)
  adj->import_csr(nrows, ncols, nvals, new_ids_ptr, new_vals_ptr,
                  new_offset_ptr);
  free(adj_offset_ptr);
  free(adj_ids_ptr);
  free(adj_vals_ptr);
}

/***
 *
 * @tparam DM
 * @param src
 * @param perm - Has the index of where to get the current (i.e. from index)
 */
template <class DM>
void rowPermuteDenseFrom(DM *&src, typename DM::itype *perm) {

  typedef typename DM::itype diT;
  typedef typename DM::ntype dnT;
  typedef typename DM::vtype dvT;

  // TODO create a copy of the current values, and then shift using the values
  // in the copied array to the new
  diT input_nrows = src->nrows();
  diT input_ncols = src->ncols();
  dvT *input_old_vals =
      (dvT *)malloc((input_ncols * input_nrows) * sizeof(dvT));
  dvT *input_vals_ptr = src->vals_ptr();

  std::copy(src->vals_ptr(), src->vals_ptr() + (input_ncols * input_nrows),
            input_old_vals);

  for (diT i_i = 0; i_i < input_nrows; i_i++) {
    dnT new_row_offset = i_i * input_ncols;
    dnT old_row_offset = perm[i_i] * input_ncols;

    for (diT i_j = 0; i_j < input_ncols; i_j++) {
      input_vals_ptr[new_row_offset + i_j] =
          input_old_vals[old_row_offset + i_j];
    }
  }
  free(input_old_vals);
}

#ifdef PR_0
/***
 *
 * @tparam DM
 * @param src
 * @param perm - Has the index of where to move the current (i.e. to index)
 */
template <class DM> void rowPermuteDenseTo(DM *&src, typename DM::itype *perm) {

  typedef typename DM::itype diT;
  typedef typename DM::ntype dnT;
  typedef typename DM::vtype dvT;
  // TODO create a copy of the current values, and then shift using the values
  // in the copied array to the new
  diT input_nrows = src->nrows();
  diT input_ncols = src->ncols();
  dvT *input_new_vals =
      (dvT *)aligned_alloc(64, (input_ncols * input_nrows) * sizeof(dvT));
  dvT *input_vals_ptr = src->vals_ptr();

#pragma omp parallel for schedule(static, 4)
  for (diT i_i = 0; i_i < input_nrows; i_i++) {
    dnT new_row_offset = perm[i_i] * input_ncols;
    dnT old_row_offset = i_i * input_ncols;

    for (diT i_j = 0; i_j < input_ncols; i_j++) {
      input_new_vals[new_row_offset + i_j] =
          input_vals_ptr[old_row_offset + i_j];
    }
  }
//    src->clear();
#ifdef ICC_CMP
  //    std::copy(pstl::execution::par_unseq, input_new_vals, input_new_vals +
  //    (input_ncols * input_nrows), input_vals_ptr);
  std::copy(input_new_vals, input_new_vals + (input_ncols * input_nrows),
            input_vals_ptr);
#else
  std::copy(input_new_vals, input_new_vals + (input_ncols * input_nrows),
            input_vals_ptr);
#endif

  //    src->import_mtx(input_new_vals);
  //    if (input_vals_ptr != nullptr){
  //        free(input_vals_ptr);
  //    }
  free(input_new_vals);
}
#elif PR_1
/***
 * TODO need to change this to only one read while still being parallel
 * @tparam DM
 * @param src
 * @param perm - Has the index of where to move the current (i.e. to index)
 */
template <class DM> void rowPermuteDenseTo(DM *&src, typename DM::itype *perm) {

  typedef typename DM::itype diT;
  typedef typename DM::ntype dnT;
  typedef typename DM::vtype dvT;
  // TODO create a copy of the current values, and then shift using the values
  // in the copied array to the new
  diT input_nrows = src->nrows();
  diT input_ncols = src->ncols();
  dvT *input_new_vals =
      (dvT *)aligned_alloc(64, (input_ncols * input_nrows) * sizeof(dvT));
  dvT *input_vals_ptr = src->vals_ptr();

#pragma omp parallel for schedule(static, 4)
  for (diT i_i = 0; i_i < input_nrows; i_i++) {
    dnT new_row_offset = perm[i_i] * input_ncols;
    dnT old_row_offset = i_i * input_ncols;

    for (diT i_j = 0; i_j < input_ncols; i_j++) {
      input_new_vals[new_row_offset + i_j] =
          input_vals_ptr[old_row_offset + i_j];
    }
  }
//    src->clear();
#ifdef ICC_CMP
  //    std::copy(pstl::execution::par_unseq, input_new_vals, input_new_vals +
  //    (input_ncols * input_nrows), input_vals_ptr);
  std::copy(input_new_vals, input_new_vals + (input_ncols * input_nrows),
            input_vals_ptr);
#else
  std::copy(input_new_vals, input_new_vals + (input_ncols * input_nrows),
            input_vals_ptr);
#endif

  //    src->import_mtx(input_new_vals);
  //    if (input_vals_ptr != nullptr){
  //        free(input_vals_ptr);
  //    }
  free(input_new_vals);
}
#endif

template <class SM>
void getMinOneBlockPerm(std::vector<SM *> adjs, typename SM::itype *perm) {
  // TODO need to consider both -1 at end AND start
  typedef typename SM::itype iT;
  typedef typename SM::ntype nT;
  typedef typename SM::vtype vT;

  iT count = 0;

  //    std::cout << "Len: " << adjs.size() << std::endl;
  for (auto adj : adjs) {
    iT nrows = adj->nrows();

    nT *adj_offset_ptr = adj->offset_ptr();
    iT *adj_ids_ptr = adj->ids_ptr();

    for (iT i_i = 0; i_i < nrows; i_i++) {
#ifdef REV_JJ
      nT e = adj_offset_ptr[i_i];
      nT e_c = adj_offset_ptr[i_i + 1];
#else
      nT e = adj_offset_ptr[i_i + 1] - 1;
      nT e_c = adj_offset_ptr[i_i] - 1;
#endif
      if (e != e_c) {
        iT u = adj_ids_ptr[e];
        if (u == -1) {
          perm[count] = i_i;
          count += 1;
        }
      }
    }
  }
}

// Why can't the permutation be a function??
template <class SM, class DM_d, class DM_i>
void rowReorder(std::vector<SM *> adjs, DM_d *input, DM_i *train_mask,
                DM_i *valid_mask, DM_i *test_mask, DM_i *labels,
                typename SM::itype *perm) {

  // Sparse input (Graph)
  typedef typename SM::itype iT;
  typedef typename SM::ntype nT;
  typedef typename SM::vtype vT;

  typedef typename DM_d::itype ddiT;
  typedef typename DM_d::ntype ddnT;
  typedef typename DM_d::vtype ddvT;

  typedef typename DM_i::itype diiT;
  typedef typename DM_i::ntype dinT;
  typedef typename DM_i::vtype divT;

  //    auto start = get_time();
  for (SM *adj : adjs) {
    iT nrows = adj->nrows();
    iT ncols = adj->ncols();
    nT nvals = adj->nvals();

    nT *adj_offset_ptr = adj->offset_ptr();
    iT *adj_ids_ptr = adj->ids_ptr();
    vT *adj_vals_ptr = adj->vals_ptr();

    // TODO Would it be more efficient to have this in alloca??
    nT *new_offset_ptr = (nT *)malloc((nrows + 1) * sizeof(nT));
    iT *new_ids_ptr = (iT *)malloc((nvals) * sizeof(iT));
    vT *new_vals_ptr = (vT *)malloc((nvals) * sizeof(vT));

    new_offset_ptr[0] = 0;

    for (iT i = 0; i < adj->nrows(); i += 1) {
      iT new_row = perm[i];
      nT new_pre_offset = new_offset_ptr[i];

      nT old_pre_offset = adj_offset_ptr[new_row];
      nT old_post_offset = adj_offset_ptr[new_row + 1];

      new_offset_ptr[i + 1] =
          new_pre_offset + (old_post_offset - old_pre_offset);
    }

#pragma omp parallel
    for (iT i = 0; i < adj->nrows(); i += 1) {
      iT new_row = perm[i];
      nT new_pre_offset = new_offset_ptr[i];

      nT old_pre_offset = adj_offset_ptr[new_row];
      nT old_post_offset = adj_offset_ptr[new_row + 1];
      nT nneigh = old_post_offset - old_pre_offset;

      std::pair<iT, vT> local_id_vals[nneigh];
      bool has_min_one = false;

      // TODO Need a way to handle -1
      for (nT e = old_pre_offset; e < old_post_offset; e++) {
        if (adj_ids_ptr[e] == -1) {
          local_id_vals[e - old_pre_offset].first = adj_ids_ptr[e];
          has_min_one = true;
        } else {
          local_id_vals[e - old_pre_offset].first = perm[adj_ids_ptr[e]];
        }
        local_id_vals[e - old_pre_offset].second = adj_vals_ptr[e];
      }

      // TODO if reordered, then need to check first not last, irrespective of
      // jj order
      if (has_min_one) {
#ifdef REV_JJ
        std::sort(local_id_vals + 1, local_id_vals + nneigh);
#else
        std::sort(local_id_vals, local_id_vals + nneigh - 1);
#endif
      } else {
        std::sort(local_id_vals, local_id_vals + nneigh);
      }

      for (nT e = 0; e < nneigh; e++) {
        new_ids_ptr[new_pre_offset] = local_id_vals[e].first;
        new_vals_ptr[new_pre_offset] = local_id_vals[e].second;
        new_pre_offset += 1;
      }

      // TODO the section below is just for debuggin
      //            for (nT e = old_pre_offset; e < old_post_offset; e++) {
      //                new_ids_ptr[new_pre_offset] = perm[adj_ids_ptr[e]];
      //                new_vals_ptr[new_pre_offset] = adj_vals_ptr[e];
      //                new_pre_offset += 1;
      //            }
    }

    adj->import_csr(nrows, ncols, nvals, new_ids_ptr, new_vals_ptr,
                    new_offset_ptr);
    free(adj_offset_ptr);
    free(adj_ids_ptr);
    free(adj_vals_ptr);
  }
  //    auto end = get_time();
  //    std::cout << "Time for sparse: " << end - start << std::endl;
  //    start = get_time();
  rowPermuteDenseFrom<DM_d>(input, perm);
  //    end = get_time();
  rowPermuteDenseFrom<DM_i>(train_mask, perm);
  rowPermuteDenseFrom<DM_i>(valid_mask, perm);
  rowPermuteDenseFrom<DM_i>(test_mask, perm);
  rowPermuteDenseFrom<DM_i>(labels, perm);

  //    std::cout << "Time for dense: " << end - start << std::endl;
}

// Why can't the permutation be a function??
/***
 *
 * @tparam SM
 * @tparam DM_d
 * @tparam DM_i
 * @param adj
 * @param input
 * @param train_mask
 * @param valid_mask
 * @param test_mask
 * @param labels
 * @param perm - Has the index of where to get the current (i.e. from index)
 */
template <class SM, class DM_d, class DM_i>
void rowReorderFrom(SM *adj, DM_d *input, DM_i *train_mask, DM_i *valid_mask,
                    DM_i *test_mask, DM_i *labels, typename SM::itype *perm) {

  // Sparse input (Graph)
  typedef typename SM::itype iT;
  typedef typename SM::ntype nT;
  typedef typename SM::vtype vT;

  typedef typename DM_d::itype ddiT;
  typedef typename DM_d::ntype ddnT;
  typedef typename DM_d::vtype ddvT;

  typedef typename DM_i::itype diiT;
  typedef typename DM_i::ntype dinT;
  typedef typename DM_i::vtype divT;

  iT nrows = adj->nrows();
  iT ncols = adj->ncols();
  nT nvals = adj->nvals();

  nT *adj_offset_ptr = adj->offset_ptr();
  iT *adj_ids_ptr = adj->ids_ptr();
  vT *adj_vals_ptr = adj->vals_ptr();

  nT *new_offset_ptr = (nT *)aligned_alloc(64, (nrows + 1) * sizeof(nT));
  iT *new_ids_ptr = (iT *)aligned_alloc(64, (nvals) * sizeof(iT));
  vT *new_vals_ptr = (vT *)aligned_alloc(64, (nvals) * sizeof(vT));

  new_offset_ptr[0] = 0;

  for (iT i = 0; i < adj->nrows(); i += 1) {
    iT new_row = perm[i];
    nT new_pre_offset = new_offset_ptr[i];

    nT old_pre_offset = adj_offset_ptr[new_row];
    nT old_post_offset = adj_offset_ptr[new_row + 1];

    new_offset_ptr[i + 1] = new_pre_offset + (old_post_offset - old_pre_offset);
  }

  //    std::cout << "------2" << std::endl;

#pragma omp parallel for schedule(dynamic) default(none)                       \
    shared(adj, new_offset_ptr, new_ids_ptr, new_vals_ptr, perm,               \
               adj_offset_ptr, adj_ids_ptr, adj_vals_ptr)
  for (iT i = 0; i < adj->nrows(); i += 1) {
    iT new_row = perm[i];
    nT new_pre_offset = new_offset_ptr[i];

    nT old_pre_offset = adj_offset_ptr[new_row];
    nT old_post_offset = adj_offset_ptr[new_row + 1];
    nT nneigh = old_post_offset - old_pre_offset;

    std::pair<iT, vT> local_id_vals[nneigh];
    bool has_min_one = false;

    // TODO This is WRONG. You are doing for move tos with the perm[id[e]], also
    // no need of the -1s here
    for (nT e = old_pre_offset; e < old_post_offset; e++) {
      if (adj_ids_ptr[e] == -1) {
        local_id_vals[e - old_pre_offset].first = adj_ids_ptr[e];
        has_min_one = true;
      } else {
        local_id_vals[e - old_pre_offset].first = perm[adj_ids_ptr[e]];
      }
      local_id_vals[e - old_pre_offset].second = adj_vals_ptr[e];
    }

    // TODO if reordered, then need to check first not last, irrespective of jj
    // order
    if (has_min_one) {
#ifdef REV_JJ
      std::sort(local_id_vals + 1, local_id_vals + nneigh);
#else
      std::sort(local_id_vals, local_id_vals + nneigh - 1);
#endif
    } else {
      std::sort(local_id_vals, local_id_vals + nneigh);
    }

    for (nT e = 0; e < nneigh; e++) {
      new_ids_ptr[new_pre_offset] = local_id_vals[e].first;
      new_vals_ptr[new_pre_offset] = local_id_vals[e].second;
      new_pre_offset += 1;
    }

    // TODO the section below is just for debuggin
    //            for (nT e = old_pre_offset; e < old_post_offset; e++) {
    //                new_ids_ptr[new_pre_offset] = perm[adj_ids_ptr[e]];
    //                new_vals_ptr[new_pre_offset] = adj_vals_ptr[e];
    //                new_pre_offset += 1;
    //            }
  }

  adj->import_csr(nrows, ncols, nvals, new_ids_ptr, new_vals_ptr,
                  new_offset_ptr);
  free(adj_offset_ptr);
  free(adj_ids_ptr);
  free(adj_vals_ptr);

  //    std::cout << "------1" << std::endl;

  rowPermuteDenseFrom<DM_d>(input, perm);
  rowPermuteDenseFrom<DM_i>(train_mask, perm);
  rowPermuteDenseFrom<DM_i>(valid_mask, perm);
  rowPermuteDenseFrom<DM_i>(test_mask, perm);
  rowPermuteDenseFrom<DM_i>(labels, perm);
}

// Why can't the permutation be a function??
/***
 *
 * @tparam SM
 * @tparam DM_d
 * @tparam DM_i
 * @param adj
 * @param input
 * @param train_mask
 * @param valid_mask
 * @param test_mask
 * @param labels
 * @param perm - Has the index of where to move the current (i.e. to index)
 */
template <class SM, class DM_d, class DM_i>
void rowReorderTo(SM *adj, DM_d *input, DM_i *train_mask, DM_i *valid_mask,
                  DM_i *test_mask, DM_i *labels, typename SM::itype *perm) {

  // Sparse input (Graph)
  typedef typename SM::itype iT;
  typedef typename SM::ntype nT;
  typedef typename SM::vtype vT;

  typedef typename DM_d::itype ddiT;
  typedef typename DM_d::ntype ddnT;
  typedef typename DM_d::vtype ddvT;

  typedef typename DM_i::itype diiT;
  typedef typename DM_i::ntype dinT;
  typedef typename DM_i::vtype divT;

#ifdef DBG_RO
  double start_time, start_total;
  double time_init, time_sparse_serial, time_sparse_parallel, time_dense,
      time_others, time_total;
  start_time = get_time();
  start_total = get_time();
#endif

  iT nrows = adj->nrows();
  iT ncols = adj->ncols();
  nT nvals = adj->nvals();

  nT *adj_offset_ptr = adj->offset_ptr();
  iT *adj_ids_ptr = adj->ids_ptr();
  vT *adj_vals_ptr = adj->vals_ptr();

  nT *new_offset_ptr = (nT *)aligned_alloc(64, (nrows + 1) * sizeof(nT));
  iT *new_ids_ptr = (iT *)aligned_alloc(64, (nvals) * sizeof(iT));
  vT *new_vals_ptr = (vT *)aligned_alloc(64, (nvals) * sizeof(vT));

  new_offset_ptr[0] = 0;

#ifdef DBG_RO
  time_init = (get_time() - start_time);
  start_time = get_time();
#endif

#pragma omp parallel for schedule(static)
  for (iT i = 0; i < adj->nrows(); i += 1) {
    iT new_loc = perm[i];
    if (new_loc < 0 || new_loc > adj->nrows()) {
      continue;
    }

    nT old_pre_offset = adj_offset_ptr[i];
    nT old_post_offset = adj_offset_ptr[i + 1];

    new_offset_ptr[new_loc + 1] = (old_post_offset - old_pre_offset);
  }

  // TODO this section can also parallelized using a parallel pre sum
  for (iT i = 0; i < adj->nrows(); i += 1) {
    new_offset_ptr[i + 1] = new_offset_ptr[i] + new_offset_ptr[i + 1];
  }

#ifdef DBG_RO
  time_sparse_serial = (get_time() - start_time);
  start_time = get_time();
#endif

#pragma omp parallel for schedule(dynamic) default(none)                       \
    shared(adj, new_offset_ptr, new_ids_ptr, new_vals_ptr, perm,               \
               adj_offset_ptr, adj_ids_ptr, adj_vals_ptr)
  for (iT i = 0; i < adj->nrows(); i += 1) {
    iT new_loc = perm[i];
    if (new_loc < 0 || new_loc > adj->nrows()) {
      continue;
    }
    nT new_pre_offset = new_offset_ptr[new_loc];

    nT old_pre_offset = adj_offset_ptr[i];
    nT old_post_offset = adj_offset_ptr[i + 1];
    nT nneigh = old_post_offset - old_pre_offset;

    std::pair<iT, vT> *local_id_vals =
        (std::pair<iT, vT> *)malloc((nneigh) * sizeof(std::pair<iT, vT>));

    for (nT e = old_pre_offset; e < old_post_offset; e++) {
      local_id_vals[e - old_pre_offset].first = perm[adj_ids_ptr[e]];
      local_id_vals[e - old_pre_offset].second = adj_vals_ptr[e];
    }

    std::sort(local_id_vals, local_id_vals + nneigh);

    for (nT e = 0; e < nneigh; e++) {
      new_ids_ptr[new_pre_offset] = local_id_vals[e].first;
      new_vals_ptr[new_pre_offset] = local_id_vals[e].second;
      new_pre_offset += 1;
    }
  }

#ifdef DBG_RO
  time_sparse_parallel = (get_time() - start_time);
#endif

  adj->import_csr(nrows, ncols, nvals, new_ids_ptr, new_vals_ptr,
                  new_offset_ptr);
  free(adj_offset_ptr);
  free(adj_ids_ptr);
  free(adj_vals_ptr);

#ifdef DBG_RO
  start_time = get_time();
#endif

  rowPermuteDenseTo(input, perm);

#ifdef DBG_RO
  time_dense = (get_time() - start_time);
  start_time = get_time();
#endif

  rowPermuteDenseTo(train_mask, perm);
  rowPermuteDenseTo(valid_mask, perm);
  rowPermuteDenseTo(test_mask, perm);
  rowPermuteDenseTo(labels, perm);

#ifdef DBG_RO
  time_others = (get_time() - start_time);
  time_total = (get_time() - start_total);

  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Debug times of reorder function" << std::endl;
  std::cout << "Init: " << time_init
            << " , percen: " << time_init * 100 / time_total << std::endl;
  std::cout << "Sparse serial: " << time_sparse_serial
            << " , percen: " << time_sparse_serial * 100 / time_total
            << std::endl;
  std::cout << "Sparse parallel: " << time_sparse_parallel
            << " , percen: " << time_sparse_parallel * 100 / time_total
            << std::endl;
  std::cout << "Dense: " << time_dense
            << " , percen: " << time_dense * 100 / time_total << std::endl;
  std::cout << "Others: " << time_others
            << " , percen: " << time_others * 100 / time_total << std::endl;
#endif
}

// Why can't the permutation be a function??
/***
 *
 * @tparam SM
 * @tparam DM
 * @tparam DL
 * @tparam DB
 * @param adj
 * @param input
 * @param train_mask
 * @param valid_mask
 * @param test_mask
 * @param labels
 * @param perm - Has the index of where to move the current (i.e. to index)
 */
template <class SM, class DM, class DL, class DB>
void rowReorderToTorch(SM *adj, DM *input, DB *train_mask, DB *valid_mask,
                       DB *test_mask, DL *labels,
                       typename SM::itype *perm) {

  // Sparse input (Graph)
  typedef typename SM::itype iT;
  typedef typename SM::ntype nT;
  typedef typename SM::vtype vT;

  typedef typename DM::itype miT;
  typedef typename DM::ntype mnT;
  typedef typename DM::vtype mvT;

  typedef typename DL::itype liT;
  typedef typename DL::ntype lnT;
  typedef typename DL::vtype lvT;

  typedef typename DB::itype biT;
  typedef typename DB::ntype bnT;
  typedef typename DB::vtype bvT;

#ifdef DBG_RO
  double start_time, start_total;
  double time_init, time_sparse_serial, time_sparse_parallel, time_dense,
      time_others, time_total;
  start_time = get_time();
  start_total = get_time();
#endif

  iT nrows = adj->nrows();
  iT ncols = adj->ncols();
  nT nvals = adj->nvals();

  nT *adj_offset_ptr = adj->offset_ptr();
  iT *adj_ids_ptr = adj->ids_ptr();
  vT *adj_vals_ptr = adj->vals_ptr();

  nT *new_offset_ptr = (nT *)aligned_alloc(64, (nrows + 1) * sizeof(nT));
  iT *new_ids_ptr = (iT *)aligned_alloc(64, (nvals) * sizeof(iT));
  vT *new_vals_ptr = (vT *)aligned_alloc(64, (nvals) * sizeof(vT));

  new_offset_ptr[0] = 0;

#ifdef DBG_RO
  time_init = (get_time() - start_time);
  start_time = get_time();
#endif

#pragma omp parallel for schedule(static)
  for (iT i = 0; i < adj->nrows(); i += 1) {
    iT new_loc = perm[i];
    if (new_loc < 0 || new_loc > adj->nrows()) {
      continue;
    }

    nT old_pre_offset = adj_offset_ptr[i];
    nT old_post_offset = adj_offset_ptr[i + 1];

    new_offset_ptr[new_loc + 1] = (old_post_offset - old_pre_offset);
  }

  // TODO this section can also parallelized using a parallel pre sum
  for (iT i = 0; i < adj->nrows(); i += 1) {
    new_offset_ptr[i + 1] = new_offset_ptr[i] + new_offset_ptr[i + 1];
  }

#ifdef DBG_RO
  time_sparse_serial = (get_time() - start_time);
  start_time = get_time();
#endif

#pragma omp parallel for schedule(dynamic) default(none)                       \
    shared(adj, new_offset_ptr, new_ids_ptr, new_vals_ptr, perm,               \
               adj_offset_ptr, adj_ids_ptr, adj_vals_ptr)
  for (iT i = 0; i < adj->nrows(); i += 1) {
    iT new_loc = perm[i];
    if (new_loc < 0 || new_loc > adj->nrows()) {
      continue;
    }
    nT new_pre_offset = new_offset_ptr[new_loc];

    nT old_pre_offset = adj_offset_ptr[i];
    nT old_post_offset = adj_offset_ptr[i + 1];
    nT nneigh = old_post_offset - old_pre_offset;

    std::pair<iT, vT> *local_id_vals =
        (std::pair<iT, vT> *)malloc((nneigh) * sizeof(std::pair<iT, vT>));

    for (nT e = old_pre_offset; e < old_post_offset; e++) {
      local_id_vals[e - old_pre_offset].first = perm[adj_ids_ptr[e]];
      local_id_vals[e - old_pre_offset].second = adj_vals_ptr[e];
    }

    std::sort(local_id_vals, local_id_vals + nneigh);

    for (nT e = 0; e < nneigh; e++) {
      new_ids_ptr[new_pre_offset] = local_id_vals[e].first;
      new_vals_ptr[new_pre_offset] = local_id_vals[e].second;
      new_pre_offset += 1;
    }
  }

#ifdef DBG_RO
  time_sparse_parallel = (get_time() - start_time);
#endif

  adj->import_csr(nrows, ncols, nvals, new_ids_ptr, new_vals_ptr,
                  new_offset_ptr);
  free(adj_offset_ptr);
  free(adj_ids_ptr);
  free(adj_vals_ptr);

#ifdef DBG_RO
  start_time = get_time();
#endif

  rowPermuteDenseTo(input, perm);

#ifdef DBG_RO
  time_dense = (get_time() - start_time);
  start_time = get_time();
#endif

  rowPermuteDenseTo(train_mask, perm);
  rowPermuteDenseTo(valid_mask, perm);
  rowPermuteDenseTo(test_mask, perm);
  rowPermuteDenseTo(labels, perm);

#ifdef DBG_RO
  time_others = (get_time() - start_time);
  time_total = (get_time() - start_total);

  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Debug times of reorder function" << std::endl;
  std::cout << "Init: " << time_init
            << " , percen: " << time_init * 100 / time_total << std::endl;
  std::cout << "Sparse serial: " << time_sparse_serial
            << " , percen: " << time_sparse_serial * 100 / time_total
            << std::endl;
  std::cout << "Sparse parallel: " << time_sparse_parallel
            << " , percen: " << time_sparse_parallel * 100 / time_total
            << std::endl;
  std::cout << "Dense: " << time_dense
            << " , percen: " << time_dense * 100 / time_total << std::endl;
  std::cout << "Others: " << time_others
            << " , percen: " << time_others * 100 / time_total << std::endl;
#endif
}

// Why can't the permutation be a function??
/***
 *
 * @tparam SM
 * @tparam DM_d
 * @tparam DM_i
 * @param adj
 * @param perm - Has the index of where to move the current (i.e. to index)
 */
template <class SM> void rowReorderToAdj(SM *adj, typename SM::itype *perm) {

  // Sparse input (Graph)
  typedef typename SM::itype iT;
  typedef typename SM::ntype nT;
  typedef typename SM::vtype vT;

  //    auto start = get_time();
  iT nrows = adj->nrows();
  iT ncols = adj->ncols();
  nT nvals = adj->nvals();

  nT *adj_offset_ptr = adj->offset_ptr();
  iT *adj_ids_ptr = adj->ids_ptr();
  vT *adj_vals_ptr = adj->vals_ptr();

  nT *new_offset_ptr = (nT *)aligned_alloc(64, (nrows + 1) * sizeof(nT));
  iT *new_ids_ptr = (iT *)aligned_alloc(64, (nvals) * sizeof(iT));
  vT *new_vals_ptr = (vT *)aligned_alloc(64, (nvals) * sizeof(vT));

  new_offset_ptr[0] = 0;

  for (iT i = 0; i < adj->nrows(); i += 1) {
    iT new_loc = perm[i];
    if (new_loc < 0 || new_loc > nrows) {
      std::cout << "A proper location was not generated for: " << i
                << std::endl;
      continue;
    }

    nT old_pre_offset = adj_offset_ptr[i];
    nT old_post_offset = adj_offset_ptr[i + 1];

    new_offset_ptr[new_loc + 1] = (old_post_offset - old_pre_offset);
  }
  for (iT i = 0; i < adj->nrows(); i += 1) {
    new_offset_ptr[i + 1] = new_offset_ptr[i] + new_offset_ptr[i + 1];
  }

#pragma omp parallel for schedule(dynamic)
  for (iT i = 0; i < adj->nrows(); i += 1) {
    iT new_loc = perm[i];
    if (new_loc < 0 || new_loc > adj->nrows()) {
      continue;
    }
    nT new_pre_offset = new_offset_ptr[new_loc];

    nT old_pre_offset = adj_offset_ptr[i];
    nT old_post_offset = adj_offset_ptr[i + 1];
    nT nneigh = old_post_offset - old_pre_offset;

    std::pair<iT, vT> *local_id_vals =
        (std::pair<iT, vT> *)malloc((nneigh) * sizeof(std::pair<iT, vT>));

    for (nT e = old_pre_offset; e < old_post_offset; e++) {
      local_id_vals[e - old_pre_offset].first = perm[adj_ids_ptr[e]];
      local_id_vals[e - old_pre_offset].second = adj_vals_ptr[e];
    }

    std::sort(local_id_vals, local_id_vals + nneigh);

    for (nT e = 0; e < nneigh; e++) {
      new_ids_ptr[new_pre_offset] = local_id_vals[e].first;
      new_vals_ptr[new_pre_offset] = local_id_vals[e].second;
      new_pre_offset += 1;
    }
  }

  adj->import_csr(nrows, ncols, nvals, new_ids_ptr, new_vals_ptr,
                  new_offset_ptr);
  free(adj_offset_ptr);
  free(adj_ids_ptr);
  free(adj_vals_ptr);
}

/**
 *
 * @tparam SM
 * @param adj
 * @param perm - This need not be the full set of columns.
 * Columns not listed will be in previously arranged order AFTER the columns
 * listed
 */
template <class SM>
void colReorder(SM *adj, typename SM::itype perm_len,
                typename SM::itype *perm) {
  // Sparse input (Graph)
  typedef typename SM::itype iT;
  typedef typename SM::ntype nT;
  typedef typename SM::vtype vT;

  iT nrows = adj->nrows();
  iT ncols = adj->ncols();
  nT nvals = adj->nvals();

  nT *adj_offset_ptr = adj->offset_ptr();
  iT *adj_ids_ptr = adj->ids_ptr();
  vT *adj_vals_ptr = adj->vals_ptr();

  // TODO Would it be more efficient to have this in alloca??
  iT *new_ids_ptr = (iT *)malloc((nvals) * sizeof(iT));
  vT *new_vals_ptr = (vT *)malloc((nvals) * sizeof(vT));

#pragma omp parallel
  for (iT i = 0; i < adj->nrows(); i += 1) {
    nT pre_offset = adj_offset_ptr[i];
    nT post_offset = adj_offset_ptr[i + 1];
    // TODO You can put this into the permutation generation function?
    //  but this would make the checks much bigger
    std::vector<bool> row_no_changes;
    for (nT init_bool = pre_offset; init_bool < post_offset; init_bool++) {
      row_no_changes.push_back(true);
    }

    iT count_added = 0;
    for (iT perm_i = 0; perm_i < perm_len; perm_i += 1) {
      for (nT e = pre_offset; e < post_offset; e++) {
        iT considered_id = adj_ids_ptr[e];
        if (considered_id == perm[perm_i]) {
          iT new_i = pre_offset + count_added;
          new_ids_ptr[new_i] = considered_id;
          new_vals_ptr[new_i] = adj_vals_ptr[e];
          row_no_changes.at(e - pre_offset) = false;
          count_added += 1;
          break;
        }
      }
    }
    for (nT e = pre_offset; e < post_offset; e++) {
      if (row_no_changes.at(e - pre_offset)) {
        iT new_i = pre_offset + count_added;
        new_ids_ptr[new_i] = adj_ids_ptr[e];
        new_vals_ptr[new_i] = adj_vals_ptr[e];
        count_added += 1;
      }
    }
  }

  // TODO change this to matrix import (from csr)
  adj->import_csr(nrows, ncols, nvals, new_ids_ptr, new_vals_ptr,
                  adj_offset_ptr);
  free(adj_ids_ptr);
  free(adj_vals_ptr);
}

template <class SM> void getAcendingOrder(SM *adj, typename SM::itype *ret) {
  typedef typename SM::itype iT;

  iT nrows = adj->nrows();

  for (iT i = 0; i < adj->nrows(); i += 1) {
    ret[i] = i;
  }
}

template <class SM> void getDecendingOrder(SM *adj, typename SM::itype *ret) {
  typedef typename SM::itype iT;

  iT nrows = adj->nrows();

  for (iT i = 0; i < adj->nrows(); i += 1) {
    ret[adj->nrows() - 1 - i] = i;
  }
}

#endif // SPARSE_ACCELERATOR_REORDERING_H
