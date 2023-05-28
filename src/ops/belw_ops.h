//Binary element wise operators.

#ifndef BELW_OPS_H
#define BELW_OPS_H

template<class vT>
inline vT sum(vT val1, vT val2){
    return val1+val2;
}

template<class vT>
inline vT mul(vT val1, vT val2){
    return val1*val2;
}

template<class vT, class dvT>
inline vT wsum(vT val1, vT val2, dvT weight){
    return val1+val2*weight;
}

template<class vT>
inline vT div(vT val1, vT val2){
    // TODO SAND do we need to add zero checks here?
    return val1/val2;
}


#endif