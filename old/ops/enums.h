#ifndef ENUMS_H
#define ENUMS_H

enum SpmmVariation {
    classic_version,
    generalized_version,
    mkl_version,
    feat_tiled_version
};

enum GcnOpsOrder{
    spmm_first,
    gemm_first
};

GcnOpsOrder get_order(int val){
    if (val == 0) return spmm_first;
    else return gemm_first;
}

SpmmVariation get_variation(int val){
    if (val == 0) return classic_version;
    if (val == 1) return generalized_version;
    if (val == 2) return mkl_version;
    else          return feat_tiled_version;
}

#endif