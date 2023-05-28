//
// Created by damitha on 4/18/22.
//

#ifndef SPARSE_ACCELERATOR_SCHEDULER_H
#define SPARSE_ACCELERATOR_SCHEDULER_H

#include <iostream>
#include <stdlib.h>

template<class SM>
void schedule_each_tile_in_vPID(std::vector<std::vector<typename SM::ntype>> &res,
                                std::vector<std::vector<typename SM::itype>> graph_row_offsets,
                                std::vector<SM*> &adjs,
                                bool allow_barriers) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;

    // Get the maximum possible rw IDs
    iT max_row_tiles = 0;
    for (iT tc_i = 0; tc_i < graph_row_offsets.size(); tc_i++) {
        iT nrows = graph_row_offsets.at(tc_i).size() - 1;
        if (max_row_tiles < nrows){
            max_row_tiles = nrows;
        }
    }
    // Init vectors
    for (iT tr_i = 0; tr_i < max_row_tiles; tr_i++){
        res.push_back(std::vector<nT>());
    }

    nT tile_count = 0;
    for (iT tc_i = 0; tc_i < graph_row_offsets.size(); tc_i++) {
        std::vector<iT> adj_row_tiles = graph_row_offsets.at(tc_i);
        for (iT tr_i = 0; tr_i < adj_row_tiles.size()-1; tr_i++) {
            res.at(tr_i).push_back(tile_count++);
        }
        if (allow_barriers){
            for (iT tr_i = 0; tr_i < max_row_tiles; tr_i++){
                res.at(tr_i).push_back((int)(tc_i + 1) * (-1));
            }
        }
    }
}

#endif //SPARSE_ACCELERATOR_SCHEDULER_H
