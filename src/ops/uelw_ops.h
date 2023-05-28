//Unary element wise operators.

#ifndef UELW_OPS_H
#define UELW_OPS_H

template<class vT>
inline vT lrelu(vT val){
    if (val < 0) {
        return 0.01 * val;
    } else {
        return val;
    }
}

template<class vT>
inline vT relu(vT val){
    if (val < 0) {
        return 0;
    } else {
        return val;
    }
}

template<class vT>
inline vT explrelu(vT val){
    return exp(lrelu<vT>(val));
}

template<class vT>
inline vT inverse_root(vT val){
    // Nodes should have at least self connection so shouldn't have 0 degrees
    return sqrt(1/val);
}


#endif