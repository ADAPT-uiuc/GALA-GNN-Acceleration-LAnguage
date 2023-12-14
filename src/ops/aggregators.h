
#include <omp.h>

//Customizable vector aggregators for gSpMM.
//The SpMM aggregators can be redefined at a set and not pair level.
//Each aggregator is subject to individual optimizations but feature-level parallelism 
//is a common notion (although LSTM aggregator perplexes the situation).

#ifndef AGGREGATORS_H
#define AGGREGATORS_H

template<class dvT, class vT, class dnT>
inline void wsumAgg(dvT *accum, dvT *to_add, vT weight, dnT length) {
    //Weighted sum aggregator.
    // TODO try moving the aggregator, to a variable and then writing back once it's finished
#ifdef LO_K_P
#pragma omp parallel for schedule(static)
    for (dnT jj = 0; jj < length; jj+=16) {
        dnT mx_j = std::min(jj+16, length);
#pragma omp simd
        for (dnT j = jj; j < mx_j; j++) {
            accum[j] += weight * to_add[j];
        }
    }
#else
//#pragma omp simd
    for (dnT j = 0; j < length; j++) {
        accum[j] += weight * to_add[j];
    }
#endif
}


template<class dvT, class vT, class dnT>
inline void maxAgg(dvT *accum, dvT *to_add, vT weight, dnT length) {
    //Max aggregator.

#pragma omp simd
    for (dnT j = 0; j < length; j++) {
        accum[j] = accum[j] >= to_add[j] ? accum[j] : to_add[j];
    }

}

template<class dvT, class dnT>
inline void sumAgg(dvT *accum, dvT *to_add, dnT length) {
    //Sum aggregator.

#pragma omp simd
    for (dnT j = 0; j < length; j++) {
        accum[j] += to_add[j];
    }
}

#endif