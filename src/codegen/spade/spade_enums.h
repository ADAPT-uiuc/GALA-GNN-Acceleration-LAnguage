//
// Created by damitha on 4/18/22.
//

#ifndef SPARSE_ACCELERATOR_BASIC_H
#define SPARSE_ACCELERATOR_BASIC_H

#include <limits>
#include <set>

enum SpInst {
    SpMM = 0,
    SDDMM = 1
};

enum MemHints {
    L0 = -1,
    L1 = 0,
    L2 = 1,
    Memory = 2
};

#endif //SPARSE_ACCELERATOR_BASIC_H
