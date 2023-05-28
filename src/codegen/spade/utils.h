//
// Created by damitha on 5/4/22.
//

#ifndef SPARSE_ACCELERATOR_UTILS_H
#define SPARSE_ACCELERATOR_UTILS_H

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include "spade_enums.h"

template<class SM>
void get_tile_cols_nnzs(std::vector<SM *> &tiled_adj,
                        std::vector<std::vector<typename SM::itype>> &graph_row_tile_offsets,
                        std::vector<typename SM::ntype> &nnz_of_tile,
                        std::vector<typename SM::itype> &col_of_tile) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    // Number of tiles
    unsigned long ntiles = 0;
    for (int r_i = 0; r_i < graph_row_tile_offsets.size(); r_i++) {
        ntiles += (graph_row_tile_offsets.at(r_i).size() - 1);
    }

    nT global_offset = 0;
    for (int t_i = 0; t_i < tiled_adj.size(); t_i++) {
        SM *local_adj = tiled_adj.at(t_i);
        std::vector<iT> local_tile_offset = graph_row_tile_offsets.at(t_i);

        iT current_row_tile = 0;
        nT prev_offset = 0;
        for (iT i = 0; i < local_adj->nrows(); i++) {
            // Print Tile start offsets to metadata
            if (i == local_tile_offset.at(current_row_tile)) {
                current_row_tile++;
                if (i != 0) {
                    nnz_of_tile.push_back(local_adj->offset_ptr()[i] - prev_offset);
                    col_of_tile.push_back(t_i);
                }
                prev_offset = local_adj->offset_ptr()[i];
            }
        }
        nnz_of_tile.push_back((local_adj->offset_ptr()[local_adj->nrows() - 1]) - prev_offset);
        col_of_tile.push_back(t_i);
        global_offset += local_adj->offset_ptr()[local_adj->nrows()];
    }
}

template<class nT>
void filter_empty_tiles(std::vector<std::vector<nT>> &schedule,
                         std::vector<nT> &nnz_of_tiles){
    for (auto &vpd_sched: schedule){
//        for (auto t: vpd_sched){
//            std::cout << t << ", ";
//        }
//        std::cout << std::endl;
//
//        std::cout << nnz_of_tiles.at(0) << " " << nnz_of_tiles.at(1) << std::endl;
//
//        for (auto t: vpd_sched){
//            if ((int)t>=0){
//                std::cout << t << std::endl;
//                std::cout << nnz_of_tiles.at(t) << ", ";
//            }
//        }
//        std::cout << std::endl;

        std::vector<nT> to_remove;
//        for (nT j = 0; j < vpd_sched.size(); j++){
//            nT t = vpd_sched.at(j);
////            std::cout  << (int)t << std::endl;
//            if ((int)t >= 0 && nnz_of_tiles.at(t) == 0){
//                to_remove.push_back(t);
//            }
//        }

        for (auto t : vpd_sched){
//            std::cout  << (int)t << std::endl;
            if ((int)t >= 0 && nnz_of_tiles.at(t) == 0){
                to_remove.push_back(t);
            }
        }

//        std::cout << "Here" << std::endl;
        for (auto t_d: to_remove){
            vpd_sched.erase(std::find(vpd_sched.begin(), vpd_sched.end(), t_d));
        }

//        std::cout << "Here" << std::endl;
//        for (auto t: vpd_sched){
//            std::cout << t << ", ";
//        }
//        std::cout << std::endl;
//
//        std::cout << "*************************" << std::endl;
    }
}


#endif //SPARSE_ACCELERATOR_UTILS_H
