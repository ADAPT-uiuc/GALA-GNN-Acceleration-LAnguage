#include <omp.h>

#include "../matrix/csrc_matrix.h"
#include "../matrix/dense_matrix.h"
#include "../utils/threading_utils.h"
#include "../ops/sparse_matrix_ops.h"
#include "../ops/dense_matrix_ops.h"
#include "../ops/aggregators.h"
#include "../ops/uelw_ops.h"
#include "../ops/belw_ops.h"
#include "../ops/samplers.h"

template<class SM, class DM>
void sage_mean_forward_layer(SM * adj, DM* h_in,
                             DM* w, DM* h_out, int support,
                             DM* h_temp, bool refresh_and_print){
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;

    // TODO create sadj as CSR or D-RM matrix
    sample_adj<SM>(adj, support, &sadj);

    // TODO Can do this as, a. create a ones function for this as well OR have a weighted sum with a ones vertex.
    gSpMM(sadj, h_in, h_temp, sum_aggregator);

    // TODO create mean / means as a vecotr or add a full on matrix boardcast (GER). Simillar issue as with above.
    auto division_operator = div<vT>;
    DVCBM(h_temp, means, division_operator);

    MM_mkl(h_temp, w, h_out);

    auto relu_operator = relu<vT>;
    UEwD<DM>(h_out,h_out,relu_operator);

    // TODO for normalize you need the size? Split into aggregation into dense then Row-wise broadcast operation?
    UEwD(h_out, h_out, normalize_operator);

    // static double upd_time = 0.0;
    // static double wh1_time = 0.0;
    // static double wh2_time = 0.0;
    // static double edge_time = 0.0;
    // static double spmm_time = 0.0;
    // static double elw_time = 0.0;

    // double t0 = get_time();

    // double t1 = get_time();
    // upd_time += t1 - t0;

    // double t2 = get_time();

    // double t3 = get_time();

    // wh1_time += t2 - t1;
    // wh2_time += t3 - t2;

    // if(fused_edge_proc){

    //     fusedGatEdgeKernel<DM, SM>(adj, wh1, wh2, a_temp);

    // }
    // else{
    //     //gSDDVV
    //     auto sum_operator = sum<vT>;
    //     gSDDVV(adj, wh1, wh2,a_temp,sum_operator);


    //     //Apply Edge = Unary ElementWise Operation on Sparse Matrix.
    //     auto explrelu_operator = explrelu<vT> ;
    //     UEwS(a_temp,a_temp,explrelu_operator);


    //     //SpMV_ones (special case- I don't express it with generalized for extra efficiency).
    //     DM sums;
    //     sums.build(h_in->nrows(),1,h_in->type());
    //     SpMV_ones(a_temp, &sums);



    //     //gSDDVV (maybe not best formulation)
    //     #pragma omp parallel for 
    //     for (iT v=0; v<adj->nrows(); v++){
    //         vT sum = 0;
    //         for(nT e=adj->offset_ptr()[v]; e<adj->offset_ptr()[v+1]; e++) {
    //             a_temp->vals_ptr()[e] = a_temp->vals_ptr()[e] / sums.vals_ptr()[(nT) v] ;
    //         }
    //     }


    // }
    // double t4=get_time();


    // edge_time += t4 - t3;

    // /// Node processing
    // //Defusing the gSpMM kernel and the Dense element wise Relu yielded acceleration.

    // auto wsum_aggr = wsumAgg<dvT,vT,dnT> ;
    // gSpMM<DM,SM>(a_temp, h_temp, h_out, wsum_aggr);

    // double t5=get_time();

    // spmm_time += t5 - t4;

    // //Unary element-wise operation on Dense Matrix: ReLU
    // auto relu_operator = relu<vT>;
    // UEwD<DM>(h_out,h_out,relu_operator);

    // double t6=get_time();

    // elw_time += t6 - t5;


    // if(refresh_and_print){
    //     std::cout<<"Dense Update took: "   <<upd_time<<std::endl;
    //     std::cout<<"Dense WH1 calculation took: "<<wh1_time<<std::endl;
    //     std::cout<<"Dense WH2 calculation took: "<<wh2_time<<std::endl;
    //     std::cout<<"Edge processing took: "<<edge_time<<std::endl;
    //     std::cout<<"SpMM took: "<<spmm_time<<std::endl;
    //     std::cout<<"Element wise ReLU on Dense took: "<<elw_time<<std::endl;
    // }
}

template<class SM, class DM>
void sage_mean_forward_layer(SM * adj, DM* h_in,
                             DM* w_self, DM* w_neigh, DM* h_out, int support,
                             DM* h_temp, DM* h_temp2, DM* w_pool, DM* bias_pool, bool refresh_and_print){
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM::itype diT;
    typedef typename DM::ntype dnT;
    typedef typename DM::vtype dvT;


    // TODO create sadj as CSR or D-RM matrix
    sample_adj<SM>(adj, support, &sadj);

    MM_mkl(h_in, w_pool, h_temp);
    DVCBM(h_temp, bias_pool, sum_operator);

    // Or another non-liner function
    auto relu_operator = relu<vT>;
    UEwD<DM>(h_temp, h_temp,relu_operator);

    gSpMM(sadj, h_in, h_temp, max_aggregator);

    MM_mkl(h_in, w_self, h_out);
    MM_mkl(h_temp, w_neigh, h_temp2);

    // TODO - Some function to add h_out and h_temp2

    auto relu_operator = relu<vT>;
    UEwD<DM>(h_out,h_out,relu_operator);

    UEwD(h_out, h_out, normalize_operator);

}