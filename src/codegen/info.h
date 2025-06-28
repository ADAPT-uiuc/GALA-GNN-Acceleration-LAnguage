//

//

#ifndef GNN_ACCELERATION_LANGUAGE_INFO_H
#define GNN_ACCELERATION_LANGUAGE_INFO_H

/// Enums
// Aggregate Structure (Array or Vector)
enum AggregateStruct{
    array,
    vector
};
AggregateStruct get_struct(int val){
    if (val == 0) return array;
    else return vector;
}
// Aggregate is directly indexed or indirectly
enum IndexType{
    direct,
    indirect
};
IndexType get_index_type(int val){
    if (val == 0) return direct;
    else return indirect;
}

/// Code-generation information
// What TACO does, is to either have compressed or uncompressed.
// If compressed then with indirection, else it's regular
template<class GM>
struct GNOpIter {
protected:
    AggregateStruct arrOrVec;
    IndexType indxType;


public:

};

template<class SM, class DM>
struct GNOpKernel {
protected:
    // Sparse input (Graph)
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    // DM Dense input / output (NN)
    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;
public:
    // Rows to get from the sparse matrix
    iT srows_start;
    iT srows_end; // Not inclusive
    // Rows to get from the dense matrix
#ifndef ST_0
    diT drows_start;
    diT drows_end; // Not inclusive
#endif
};


#endif //GNN_ACCELERATION_LANGUAGE_INFO_H
