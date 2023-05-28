//
// Created by damitha on 4/20/22.
//
#include <iostream>
#include <fstream>
#include <stdlib.h>

#ifndef SPARSE_ACCELERATOR_PRINT_FILES_H
#define SPARSE_ACCELERATOR_PRINT_FILES_H

#include "spade_enums.h"

void print_op_config(const std::string &out_path,
                     int level,
                     bool prefetch,
                     bool barriers,
                     bool bypass_caches) {
    // Init data and metadata files
    std::ofstream ops_file;
    ops_file.open(out_path + "aggregate_" + std::to_string(level) + "_ops.txt");
    // Prefetch
    ops_file << prefetch << "\n";
    // Add barriers
    ops_file << prefetch << "\n";
    // Bypass caches
    ops_file << prefetch << "\n";
    ops_file.close();
}


template<class nT, class pT>
void print_shedule(const std::string &out_path, int level, std::vector<std::vector<nT>> &schedule) {
    // Schedule the tiles to PEs
    std::ofstream sched_file;
    sched_file.open(out_path + "aggregate_" + std::to_string(level) + "_shed.txt");
    sched_file << schedule.size() << "\n";
    for (int s_i = 0; s_i < schedule.size(); s_i++) {
        std::vector<nT> row = schedule.at(s_i);
        for (int s_j = 0; s_j < row.size(); s_j++) {
            sched_file << s_i << "," << (pT) row.at(s_j) << "\n";
        }
    }
    sched_file.close();
}

/***
 * SpMM code generation with empty
 * @tparam SM
 * @tparam DM
 * @param out_path
 * @param level
 * @param Instruction
 * @param row_ids
 * @param col_ids
 * @param vals
 * @param input_emb_B
 * @param input_emb_C
 * @param new_output
 * @param leading_dims
 * @param nnzs
 * @param tiled_adj
 * @param graph_row_tile_offsets
 * @param nrows
 * @param rows_per_tile
 */
template<class SM, class DM>
void print_meta_n_data(const std::string &out_path,
                       int level,
                       SpInst Instruction,
                       typename SM::itype *row_ids,
                       typename SM::itype *col_ids,
                       typename SM::vtype *vals,
                       DM *input_emb_B,
                       DM *input_emb_C,
                       DM *new_output,
                       long leading_dims,
                       typename SM::ntype nnzs,
                       std::vector<SM *> &tiled_adj,
                       std::vector<std::vector<typename SM::itype>> &graph_row_tile_offsets,
                       typename SM::itype nrows,
                       typename SM::itype rows_per_tile) {

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    // Init data and metadata files
    std::ofstream data_file;
    std::ofstream metadata_file;
    data_file.open(out_path + "aggregate_" + std::to_string(level) + "_data.txt");
    metadata_file.open(out_path + "aggregate_" + std::to_string(level) + "_metadata.txt");
    // SpMM - 0, or SDDMM - 1
    metadata_file << Instruction << "\n"; /// Indicate that this operation in SpMM
    // data[0] - Row IDs for COO
    metadata_file << (uint64_t) row_ids
                  << "\n"; // TODO Ask how this would work? If it's fine to just have dummy values?
    // data[1] - Col IDs for COO
    metadata_file << (uint64_t) col_ids << "\n";
    // data[2] - Vals for COO
    metadata_file << (uint64_t) vals << "\n";
    // data[3] - nullptr for COO
    metadata_file << (uint64_t)
            static_cast<void *>(nullptr) << "\n";
#ifdef ACC_RB
    // B base
    metadata_file << (uint64_t) 0 << "\n";
    // C base
    metadata_file << (uint64_t) 0 << "\n";
    // D base
    metadata_file << (uint64_t) 0 << "\n";
#else
    // B base
    metadata_file << (uint64_t) input_emb_B->vals_ptr() << "\n";
    // C base
    metadata_file << (uint64_t) input_emb_C->vals_ptr() << "\n";
    // D base
    metadata_file << (uint64_t) new_output->vals_ptr() << "\n";
#endif
    // Leading dim of dense
    metadata_file << leading_dims << "\n";
    // Number of NNZs
    metadata_file << nnzs << "\n";
    // Number of nodes
    metadata_file << nrows << "\n";
    // Number of rows in a tile
    metadata_file << rows_per_tile << "\n";

    // Number of tiles
    unsigned long ntiles = 0;
    for (int r_i = 0; r_i < graph_row_tile_offsets.size(); r_i++) {
        ntiles += (graph_row_tile_offsets.at(r_i).size() - 1);
    }
    metadata_file << ntiles << "\n";

    nT global_offset = 0;
    for (int t_i = 0; t_i < tiled_adj.size(); t_i++) {
        SM *local_adj = tiled_adj.at(t_i);
        std::vector<iT> local_tile_offset = graph_row_tile_offsets.at(t_i);

        iT current_row_tile = 0;
        nT prev_offset = 0;
        for (iT i = 0; i < local_adj->nrows(); i++) {
            // Print COO to data
            for (iT j = local_adj->offset_ptr()[i]; j < local_adj->offset_ptr()[i + 1]; j++) {
                data_file << i << "," << local_adj->ids_ptr()[j] << "," << local_adj->vals_ptr()[j] << "\n";
            }
            // Print Tile start offsets to metadata
            if (i == local_tile_offset.at(current_row_tile)) {
                metadata_file << global_offset + local_adj->offset_ptr()[i] << "\n";
                current_row_tile++;
//                if (i != 0){
//                    // TODO Need to move this out of this
//                    nnz_of_tile.push_back(local_adj->offset_ptr()[i] - prev_offset);
//                    col_of_tile.push_back(t_i);
//                }
                prev_offset = local_adj->offset_ptr()[i];
            }
        }
        // TODO Need to move this out of this
//        nnz_of_tile.push_back((local_adj->offset_ptr()[local_adj->nrows()-1]) - prev_offset);
//        col_of_tile.push_back(t_i);
        global_offset += local_adj->offset_ptr()[local_adj->nrows()];
    }
    data_file.close();
    metadata_file.close();
}

/***
 * SpMM code generation without empty
 * @tparam SM
 * @tparam DM
 * @param out_path
 * @param level
 * @param Instruction
 * @param row_ids
 * @param col_ids
 * @param vals
 * @param input_emb_B
 * @param input_emb_C
 * @param new_output
 * @param leading_dims
 * @param nnzs
 * @param adj
 * @param cols_per_tile
 * @param schedule
 * @param nrows
 * @param rows_per_tile
 */
template<class SM, class DM>
void print_meta_n_data(const std::string &out_path,
                       int level,
                       SpInst Instruction,
                       typename SM::itype *row_ids,
                       typename SM::itype *col_ids,
                       typename SM::vtype *vals,
                       DM *input_emb_B,
                       DM *input_emb_C,
                       DM *new_output,
                       long leading_dims,
                       typename SM::ntype nnzs,
                       SM *adj,
                       typename SM::itype cols_per_tile,
                       std::vector<std::vector<typename SM::ntype>> &schedule,
                       typename SM::itype nrows,
                       typename SM::itype rows_per_tile) {

    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    // Init data and metadata files
    std::ofstream data_file;
    std::ofstream metadata_file;
    data_file.open(out_path + "aggregate_" + std::to_string(level) + "_data.txt");
    metadata_file.open(out_path + "aggregate_" + std::to_string(level) + "_metadata.txt");
    // SpMM - 0, or SDDMM - 1
    metadata_file << Instruction << "\n"; /// Indicate that this operation in SpMM
    // data[0] - Row IDs for COO
    metadata_file << (uint64_t) row_ids
                  << "\n"; // TODO Ask how this would work? If it's fine to just have dummy values?
    // data[1] - Col IDs for COO
    metadata_file << (uint64_t) col_ids << "\n";
    // data[2] - Vals for COO
    metadata_file << (uint64_t) vals << "\n";
    // data[3] - nullptr for COO
    metadata_file << (uint64_t)
            static_cast<void *>(nullptr) << "\n";
#ifdef ACC_RB
    // B base
    metadata_file << (uint64_t) 0 << "\n";
    // C base
    metadata_file << (uint64_t) 0 << "\n";
    // D base
    metadata_file << (uint64_t) 0 << "\n";
#else
    // B base
    metadata_file << (uint64_t) input_emb_B->vals_ptr() << "\n";
    // C base
    metadata_file << (uint64_t) input_emb_C->vals_ptr() << "\n";
    // D base
    metadata_file << (uint64_t) new_output->vals_ptr() << "\n";
#endif
    // Leading dim of dense
    metadata_file << leading_dims << "\n";
    // Number of NNZs
    metadata_file << nnzs << "\n";
    // Number of nodes
    metadata_file << nrows << "\n";
    // Number of rows in a tile
    metadata_file << rows_per_tile << "\n";

    // Number of tiles
    unsigned long ntiles = 0;

    std::vector<nT> tile_offsets;
    auto copy_offsets = (nT *) aligned_alloc(64, (adj->nrows() + 1) * sizeof(nT));
    memcpy(copy_offsets, adj->offset_ptr(), (adj->nrows() + 1) * sizeof(nT));

    nT current_val = 0;
    nT *A_offset_ptr = adj->offset_ptr();
    // Tile columns
    for (iT j = 0; j < adj->ncols(); j += cols_per_tile) {
        auto drows_end = std::min(j + cols_per_tile, adj->ncols());
        // Tile rows
        nT vPE = 0;
        for (iT i = 0; i < adj->nrows(); i += rows_per_tile) {
            auto srows_end = std::min(i + rows_per_tile, adj->nrows());

            // Track if NNZs are in the tile
            bool found_nnzs = false;
            // Calculate teh vPE to hand the execution to

            if (j == 0) {
                std::vector<nT> local_schedule;
                schedule.push_back(local_schedule);
            }

            // Iterate the rows in the tile
            for (iT v = i; v < srows_end; v++) {
                nT first_node_edge = copy_offsets[v];
                nT last_node_edge = A_offset_ptr[v + 1];

                // Iterate through edges
                for (nT e = first_node_edge; e < last_node_edge; e++) {
                    iT u = adj->ids_ptr()[e];
                    if (u >= j && u < drows_end) {
                        vT A_val = adj->vals_ptr()[e];

                        if (!found_nnzs) {
                            found_nnzs = true;
                            tile_offsets.push_back(current_val);

                            schedule.at(vPE).push_back(ntiles);

                            ntiles += 1;
                        }

//                        data_file << v << "," << u << "," << A_val << "\n";
                        data_file << v << "," << u << "\n"; // TODO values are removed
                        current_val += 1;
                    } else if (u >= drows_end) {
                        copy_offsets[v] = e;
                        break;
                    }
                }
            }
            vPE += 1;
        }
    }
    metadata_file << ntiles << "\n";

    // Print the metadata
    for (nT t_i = 0; t_i < ntiles; t_i++) {
        metadata_file << tile_offsets[t_i] << "\n";
    }
    data_file.close();
    metadata_file.close();
}

/***
 * SDDMM code generation with empty
 * @tparam SM
 * @tparam DM
 * @param out_path
 * @param level
 * @param Instruction
 * @param row_ids
 * @param col_ids
 * @param vals
 * @param input_emb_B
 * @param input_emb_C
 * @param new_output
 * @param leading_dims
 * @param nnzs
 * @param tiled_adj
 * @param graph_row_tile_offsets
 * @param nrows
 * @param rows_per_tile
 */
template<class SM, class DM>
void print_meta_n_data(const std::string &out_path,
                       int level,
                       SpInst Instruction,
                       typename SM::itype *row_ids,
                       typename SM::itype *col_ids,
                       typename SM::vtype *vals,
                       DM *input_emb_B,
                       DM *input_emb_C,
                       SM *new_output,
                       long leading_dims,
                       typename SM::ntype nnzs,
                       std::vector<SM *> &tiled_adj,
                       std::vector<std::vector<typename SM::itype>> &graph_row_tile_offsets,
                       typename SM::itype nrows,
                       typename SM::itype rows_per_tile) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;


    // Init data and metadata files
    std::ofstream data_file;
    std::ofstream metadata_file;
    data_file.open(out_path + "aggregate_" + std::to_string(level) + "_data.txt");
    metadata_file.open(out_path + "aggregate_" + std::to_string(level) + "_metadata.txt");
    // SpMM - 0, or SDDMM - 1
    metadata_file << Instruction << "\n"; /// Indicate that this operation in SpMM
    // data[0] - Row IDs for COO
    metadata_file << (uint64_t) row_ids
                  << "\n"; // TODO Ask how this would work? If it's fine to just have dummy values?
    // data[1] - Col IDs for COO
    metadata_file << (uint64_t) col_ids << "\n";
    // data[2] - Vals for COO
    metadata_file << (uint64_t) vals << "\n";
    // data[3] - nullptr for COO
    metadata_file << (uint64_t)
            static_cast<void *>(nullptr) << "\n";
#ifdef ACC_RB
    // B base
    metadata_file << (uint64_t) 0 << "\n";
    // C base
    metadata_file << (uint64_t) 0 << "\n";
    // D base
    metadata_file << (uint64_t) 0 << "\n";
#else
    // B base
    metadata_file << (uint64_t) input_emb_B->vals_ptr() << "\n";
    // C base
    metadata_file << (uint64_t) input_emb_C->vals_ptr() << "\n";
    // D base
    metadata_file << (uint64_t) new_output->vals_ptr() << "\n";
#endif
    // Leading dim of dense
    metadata_file << leading_dims << "\n";
    // Number of NNZs
    metadata_file << nnzs << "\n";
    // Number of nodes
    metadata_file << nrows << "\n";
    // Number of rows in a tile
    metadata_file << rows_per_tile << "\n";

    // Number of tiles
    unsigned long ntiles = 0;
    for (int r_i = 0; r_i < graph_row_tile_offsets.size(); r_i++) {
        ntiles += (graph_row_tile_offsets.at(r_i).size() - 1);
    }
    metadata_file << ntiles << "\n";

    nT global_offset = 0;
    for (int t_i = 0; t_i < tiled_adj.size(); t_i++) {
        SM *local_adj = tiled_adj.at(t_i);
        std::vector<iT> local_tile_offset = graph_row_tile_offsets.at(t_i);

        iT current_row_tile = 0;
        nT prev_offset = 0;
        for (iT i = 0; i < local_adj->nrows(); i++) {
            // Print COO to data
            for (iT j = local_adj->offset_ptr()[i]; j < local_adj->offset_ptr()[i + 1]; j++) {
                data_file << i << "," << local_adj->ids_ptr()[j] << "," << local_adj->vals_ptr()[j] << "\n";
            }
            // Print Tile start offsets to metadata
            if (i == local_tile_offset.at(current_row_tile)) {
                metadata_file << global_offset + local_adj->offset_ptr()[i] << "\n";
                current_row_tile++;
//                if (i != 0){
//                    // TODO Need to move this out of this
//                    nnz_of_tile.push_back(local_adj->offset_ptr()[i] - prev_offset);
//                    col_of_tile.push_back(t_i);
//                }
                prev_offset = local_adj->offset_ptr()[i];
            }
        }
        // TODO Need to move this out of this
//        nnz_of_tile.push_back((local_adj->offset_ptr()[local_adj->nrows()-1]) - prev_offset);
//        col_of_tile.push_back(t_i);
        global_offset += local_adj->offset_ptr()[local_adj->nrows()];
    }
    data_file.close();
    metadata_file.close();
}

/***
 * SDDMM code generation without empty
 * @tparam SM
 * @tparam DM
 * @param out_path
 * @param level
 * @param Instruction
 * @param row_ids
 * @param col_ids
 * @param vals
 * @param input_emb_B
 * @param input_emb_C
 * @param new_output
 * @param leading_dims
 * @param nnzs
 * @param tiled_adj
 * @param graph_row_tile_offsets
 * @param nrows
 * @param rows_per_tile
 */
template<class SM, class DM>
void print_meta_n_data(const std::string &out_path,
                       int level,
                       SpInst Instruction,
                       typename SM::itype *row_ids,
                       typename SM::itype *col_ids,
                       typename SM::vtype *vals,
                       DM *input_emb_B,
                       DM *input_emb_C,
                       SM *new_output,
                       long leading_dims,
                       typename SM::ntype nnzs,
                       SM *adj,
                       typename SM::itype cols_per_tile,
                       std::vector<std::vector<typename SM::ntype>> &schedule,
                       typename SM::itype nrows,
                       typename SM::itype rows_per_tile) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;


    // Init data and metadata files
    std::ofstream data_file;
    std::ofstream metadata_file;
    data_file.open(out_path + "aggregate_" + std::to_string(level) + "_data.txt");
    metadata_file.open(out_path + "aggregate_" + std::to_string(level) + "_metadata.txt");
    // SpMM - 0, or SDDMM - 1
    metadata_file << Instruction << "\n"; /// Indicate that this operation in SpMM
    // data[0] - Row IDs for COO
    metadata_file << (uint64_t) row_ids
                  << "\n"; // TODO Ask how this would work? If it's fine to just have dummy values?
    // data[1] - Col IDs for COO
    metadata_file << (uint64_t) col_ids << "\n";
    // data[2] - Vals for COO
    metadata_file << (uint64_t) vals << "\n";
    // data[3] - nullptr for COO
    metadata_file << (uint64_t)
            static_cast<void *>(nullptr) << "\n";
#ifdef ACC_RB
    // B base
    metadata_file << (uint64_t) 0 << "\n";
    // C base
    metadata_file << (uint64_t) 0 << "\n";
    // D base
    metadata_file << (uint64_t) 0 << "\n";
#else
    // B base
    metadata_file << (uint64_t) input_emb_B->vals_ptr() << "\n";
    // C base
    metadata_file << (uint64_t) input_emb_C->vals_ptr() << "\n";
    // D base
    metadata_file << (uint64_t) new_output->vals_ptr() << "\n";
#endif
    // Leading dim of dense
    metadata_file << leading_dims << "\n";
    // Number of NNZs
    metadata_file << nnzs << "\n";
    // Number of nodes
    metadata_file << nrows << "\n";
    // Number of rows in a tile
    metadata_file << rows_per_tile << "\n";

    // Number of tiles
    unsigned long ntiles = 0;

    std::vector<nT> tile_offsets;
    auto copy_offsets = (nT *) aligned_alloc(64, (adj->nrows() + 1) * sizeof(nT));
    memcpy(copy_offsets, adj->offset_ptr(), (adj->nrows() + 1) * sizeof(nT));

    nT current_val = 0;
    nT *A_offset_ptr = adj->offset_ptr();
    // Tile columns
    for (iT j = 0; j < adj->ncols(); j += cols_per_tile) {
        auto drows_end = std::min(j + cols_per_tile, adj->ncols());
        // Tile rows
        nT vPE = 0;
        for (iT i = 0; i < adj->nrows(); i += rows_per_tile) {
            auto srows_end = std::min(i + rows_per_tile, adj->nrows());

            // Track if NNZs are in the tile
            bool found_nnzs = false;
            // Calculate teh vPE to hand the execution to

            if (j == 0) {
                std::vector<nT> local_schedule;
                schedule.push_back(local_schedule);
            }

            // Iterate the rows in the tile
            for (iT v = i; v < srows_end; v++) {
                nT first_node_edge = copy_offsets[v];
                nT last_node_edge = A_offset_ptr[v + 1];

                // Iterate through edges
                for (nT e = first_node_edge; e < last_node_edge; e++) {
                    iT u = adj->ids_ptr()[e];
                    if (u >= j && u < drows_end) {
                        vT A_val = adj->vals_ptr()[e];

                        if (!found_nnzs) {
                            found_nnzs = true;
                            tile_offsets.push_back(current_val);

                            schedule.at(vPE).push_back(ntiles);

                            ntiles += 1;
                        }

                        data_file << v << "," << u << "," << A_val << "\n";
                        current_val += 1;
                    } else if (u >= drows_end) {
                        copy_offsets[v] = e;
                        break;
                    }
                }
            }
            vPE += 1;
        }
    }
    metadata_file << ntiles << "\n";

    // Print the metadata
    for (nT t_i = 0; t_i < ntiles; t_i++) {
        metadata_file << tile_offsets[t_i] << "\n";
    }
    data_file.close();
    metadata_file.close();
}

#endif //SPARSE_ACCELERATOR_PRINT_FILES_H
