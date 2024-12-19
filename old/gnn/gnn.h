
#include "../matrix/csrc_matrix.h"
#include "../matrix/coo_matrix.h"

#include "../matrix/dense_matrix.h"
#include "gcn_layer.h"
#include "gat_layer.h"

const int TIMES_TO_RECORD = 4;

//Separating dense matrices, since some are storing doubles and others integers.
template<class SM, class DM1, class DM2>
class GNN {
    // Simple wrapper class for GNN when used in semi-supervised node classification.
protected:
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    typedef typename DM1::itype diT1;
    typedef typename DM1::ntype dnT1;
    typedef typename DM1::vtype dvT1;

    typedef typename DM2::itype diT2;
    typedef typename DM2::ntype dnT2;
    typedef typename DM2::vtype dvT2;

    //num_layers includes input and output layer
    int num_layers;
    int num_classes;
    iT num_nodes;
    std::vector<diT1> num_neurons;


    SM *nadj = nullptr;
    //Input embeddings do not need to be stored separately from layer_acts.
    DM2 *labels;
    DM2 *train_masks;
    DM2 *valid_masks;
    DM2 *test_masks;

    // SAND
    DM1 *degrees;

    std::vector<DM1 *> weights;
    std::vector<DM1 *> biases;

    //Activations (layer outputs=inputs to next layer)
    //*(layer_acts[0]) stores input embeddings.
    std::vector<DM1 *> layer_acts;

//    //Weight gradients used in learning algorithm.
//    std::vector<DM1 *> dweights;
//    std::vector<DM1 *> dbiases;
//    //Intermediate gradients used in backpropagation.
    std::vector<DM1 *> dZ;

    double **individual_layer_times;

public:
//TODO: OTHER COMMON FUNCTIONALITIES
    void acc_evaluation() {
        //Assuming that output with maximum value corresponds to predicted class.
        //This method does not need to be optimized.
        assert(num_classes == num_neurons.back());
        int train_size = 0;
        int train_cpreds = 0;
        int test_size = 0;
        int test_cpreds = 0;
        int val_size = 0;
        int val_cpreds = 0;
        //Train set accuracy
        for (dnT1 i = 0; i < num_nodes; i++) {
            int correct_prediction = 0;
            dnT1 predicted_class = 0;
            //Check activations of final layer choose the maximum and predict class=argmax.
            dvT1 max = layer_acts[num_layers - 1]->vals_ptr()[i * num_classes + 0];
            for (dnT1 j = 0; j < num_classes; j++) {
                dvT1 new_val = layer_acts[num_layers - 1]->vals_ptr()[i * num_classes + j];
//                std::cout << new_val << " ";
                if (new_val > max) {
                    max = new_val;
                    predicted_class = j;
                }
            }
//            std::cout<<std::endl;
            ////std::cout<<labels->vals_ptr()[i]<<" "<<predicted_class<<std::endl;
            //If predicted equals ground truth.
            if (predicted_class == labels->vals_ptr()[i]) {
                correct_prediction = 1;
            }
            if (train_masks->vals_ptr()[i] == 1) {
                train_size++;
                train_cpreds += correct_prediction;
            }
            if (test_masks->vals_ptr()[i] == 1) {
                test_size++;
                test_cpreds += correct_prediction;
            }
            if (valid_masks->vals_ptr()[i] == 1) {
                val_size++;
                val_cpreds += correct_prediction;
            }

        }
//        std::cout << "Classes: " << num_neurons.back() << " ,Act layers num: " << layer_acts.size() << " ,Num layers: "
//                  << num_layers << " ,Num neurons num: " << num_neurons.size() << std::endl;
        std::cout << "Accuracy on test set:" << (float) test_cpreds / (float) test_size << " " << test_size << " "
                  << test_cpreds << std::endl;
        std::cout << "Accuracy on train set:" << (float) train_cpreds / (float) train_size << " " << train_size << " "
                  << train_cpreds << std::endl;
        std::cout << "Accuracy on validation set:" << (float) val_cpreds / (float) val_size << " " << val_size << " "
                  << val_cpreds << std::endl;
    }


};


template<class SM, class DM1, class DM2>
class GCN : public GNN<SM, DM1, DM2> {
    //GCN constructor
    //Values of constant matrices are shallow-copied!
    //Heap space is allocated for the output of each layer and the corresponding intermediate gradient.
public:
    GCN(SM *nadj_,
        DM1 *input_emb_,
        DM2 *labels_,
        DM2 *train_masks_,
        DM2 *valid_masks_,
        DM2 *test_masks_,
        std::vector<DM1 *> weights_,
        std::vector<DM1 *> biases_,
        int num_layers_) {

        this->num_nodes = nadj_->nrows();
        this->num_layers = num_layers_;

        this->num_neurons.push_back(weights_[0]->nrows());
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->num_neurons.push_back(weights_[i]->ncols());
        }

        //Shallow copies
        this->nadj = nadj_;
        this->labels = labels_;
        this->train_masks = train_masks_;
        this->valid_masks = valid_masks_;
        this->test_masks = test_masks_;
        this->num_classes = this->num_neurons.back();

        this->individual_layer_times = new double *[this->num_layers - 1];

        //Deep copies
        //We are either initializing the weights to the initialization tensorflow values or the post-training ones.

        this->layer_acts.push_back(input_emb_);

        for (int i = 0; i < this->num_layers - 1; i++) {
            //std::cout<<"Generated intermediate matrices for layer "<<i + 1<<std::endl;
            //weights[0] corresponds to weights between layers 0 and 1.
            this->individual_layer_times[i] = new double[TIMES_TO_RECORD];
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                this->individual_layer_times[i][time_i] = 0;
            }
            this->weights.push_back(weights_[i]);
            this->biases.push_back(biases_[i]);

            //Allocating dense matrix objects for layer activations-not storing any values.
            DM1 *new_layer_act = new DM1;
            //For the time being we are storing activation matrices in the form (num_neurons)x(num_nodes).
            //TODO: Change to transpose if better locality is exhibited.
#ifdef A_ALLOC
            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type(), 0);
#else
            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type());
#endif
            this->layer_acts.push_back(new_layer_act);
        }

//        // SAND - TODO This doesn't fit here. Only works if this is the first layer.
//        this->degrees = new DM1;
//        this->degrees->build(nadj_->nrows(), 1, weights_[0]->type());
//        static double norm_clc_start = get_time();
//        SpMV_ones(this->nadj, this->degrees);
//        auto inverse_root_operator = inverse_root<typename DM1::vtype>;
//        // For the degrees vector (since diagonal matrix), change their values to make D^(-1/2) by getting the inverse
//        // root of each element
//        UEwD(this->degrees, this->degrees, inverse_root_operator);
//        auto mul_operator = mul<typename DM1::vtype>;
//        // For each "row" in the sparse matrix multiply the value by the normalized degree matrix
//        // A = D^(-1/2) * A
//        SpVRBM(this->nadj, this->degrees, this->nadj, mul_operator);
//        // For each "column" in the sparse matrix multiply the value by the normalized degree matrix
//        // A = A * D^(-1/2)
//        SpVCBM(this->nadj, this->degrees, this->nadj, mul_operator);
//        static double norm_clc_end = get_time();
//
//        std::cout << "Norm calc took : " << (norm_clc_end - norm_clc_start) << std::endl;

#ifdef GN_2
        this->degrees = new DM1;
        this->degrees->build(nadj_->nrows(), 1, weights_[0]->type(), 0);
        SpMV_ones(this->nadj, this->degrees);
        auto inverse_root_operator = inverse_root<typename DM1::vtype>;
        UEwD(this->degrees, this->degrees, inverse_root_operator);
        MMbroacast_row(this->layer_acts[0], this->degrees, this->layer_acts[0]);
#endif

    }

    void forward_pass(SpmmVariation spmm_variation,
                      int tile_size,
                      bool refresh_and_print) {
        for (int i = 0; i < this->num_layers - 1; i++) {
            // SAND
            // Do update or aggregate first
            GcnOpsOrder order = gemm_first;
            if (this->weights[i]->ncols() >= this->weights[i]->nrows()) {
                order = spmm_first;
            }
#ifdef GN_2
            if (i != 0) {
                MMbroacast_row(this->layer_acts[i], this->degrees, this->layer_acts[i]);
            }
#endif
            // Forward function
            gcn_forward_layer<SM, DM1>(this->nadj,
#ifdef GN_2
                    this->degrees,
#endif
                                       this->layer_acts[i],
                                       this->weights[i],
                                       this->biases[i],
                                       this->layer_acts[i + 1],
                                       order,
                                       spmm_variation,
                                       tile_size,
                                       refresh_and_print && (i == this->num_layers - 2),
                                       i,
                                       this->individual_layer_times);
        }
        if (refresh_and_print) {
            double sums[TIMES_TO_RECORD] = {0};
            for (int i = 0; i < this->num_layers - 1; i++) {
                std::cout << "Times between layers " << i << " and " << i + 1 << ": ";
                for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                    sums[time_i] += this->individual_layer_times[i][time_i];
                    std::cout << this->individual_layer_times[i][time_i] << ",";
                    this->individual_layer_times[i][time_i] = 0;
                }
                std::cout << std::endl;
            }
            std::cout << "*****" << std::endl;
            std::cout << "Summed times of operations: ";
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                std::cout << sums[time_i] << ",";
            }
            std::cout << std::endl;
        }
    }
#ifdef MKL

    void forward_pass_mkl(SpmmVariation spmm_variation,
                          int tile_size,
                          bool refresh_and_print,
                          sparse_matrix_t *A,
                          matrix_descr descrA) {
        for (int i = 0; i < this->num_layers - 1; i++) {
            // SAND
            // Do update or aggregate first
            GcnOpsOrder order = gemm_first;
            if (this->weights[i]->ncols() >= this->weights[i]->nrows()) {
                order = spmm_first;
            }
#ifdef GN_2
            if (i != 0) {
                MMbroacast_row(this->layer_acts[i], this->degrees, this->layer_acts[i]);
            }
#endif
            // Forward function
            gcn_forward_layer_mkl<SM, DM1>(this->nadj,
#ifdef GN_2
                    this->degrees,
#endif
                                           this->layer_acts[i],
                                           this->weights[i],
                                           this->biases[i],
                                           this->layer_acts[i + 1],
                                           order,
                                           spmm_variation,
                                           tile_size,
                                           refresh_and_print && (i == this->num_layers - 2),
                                           i,
                                           this->individual_layer_times,
                                           A, descrA);
        }
        if (refresh_and_print) {
            double sums[TIMES_TO_RECORD] = {0};
            for (int i = 0; i < this->num_layers - 1; i++) {
                std::cout << "Times between layers " << i << " and " << i + 1 << ": ";
                for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                    sums[time_i] += this->individual_layer_times[i][time_i];
                    std::cout << this->individual_layer_times[i][time_i] << ",";
                    this->individual_layer_times[i][time_i] = 0;
                }
                std::cout << std::endl;
            }
            std::cout << "*****" << std::endl;
            std::cout << "Summed times of operations: ";
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                std::cout << sums[time_i] << ",";
            }
            std::cout << std::endl;
        }
    }

#endif

//    void forward_pass_tiled(SpmmVariation spmm_variation, int tile_size, bool refresh_and_print, int sps, int dns) {
//        for (int i = 0; i < this->num_layers - 1; i++) {
//            // SAND
//            // Do update or aggregate first
//            GcnOpsOrder order = gemm_first;
//            if (this->weights[i]->ncols() >= this->weights[i]->nrows()) {
//                order = spmm_first;
//            }
//            gcn_forward_layer_tiled<SM, DM1>(this->nadj,
//                                             this->layer_acts[i],
//                                             this->weights[i],
//                                             this->biases[i],
//                                             this->layer_acts[i + 1],
//                                             order,
//                                             spmm_variation,
//                                             tile_size,
//                                             refresh_and_print && (i == this->num_layers - 2),
//                                             sps, dns,
//                                             i,
//                                             this->individual_layer_times);
//        }
//        if (refresh_and_print) {
//            double sums[TIMES_TO_RECORD] = {0};
//            for (int i = 0; i < this->num_layers - 1; i++) {
//                std::cout << "Times between layers " << i << " and " << i + 1 << ": ";
//                for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
//                    sums[time_i] += this->individual_layer_times[i][time_i];
//                    std::cout << this->individual_layer_times[i][time_i] << ",";
//                }
//                std::cout << std::endl;
//            }
//            std::cout << "*****" << std::endl;
//            std::cout << "Summed times of operations: ";
//            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
//                std::cout << sums[time_i] << ",";
//            }
//            std::cout << std::endl;
//        }
//    }
//
//    void forward_pass_dgl_tiled(SpmmVariation spmm_variation, int tile_size, bool refresh_and_print, int sps, int dns) {
//        for (int i = 0; i < this->num_layers - 1; i++) {
//            // SAND
//            GcnOpsOrder order = gemm_first;
//            if (this->weights[i]->ncols() >= this->weights[i]->nrows()) {
//                order = spmm_first;
//            }
//            gcn_forward_layer_dgl_tiled<SM, DM1>(this->nadj,
//                                                 this->layer_acts[i],
//                                                 this->weights[i],
//                                                 this->biases[i],
//                                                 this->layer_acts[i + 1],
//                                                 order,
//                                                 spmm_variation,
//                                                 tile_size,
//                                                 refresh_and_print && (i == this->num_layers - 2),
//                                                 sps, dns);
//        }
//    }
//
//    void forward_pass_slice_dgl_tiled(SpmmVariation spmm_variation, int tile_size, bool refresh_and_print,
//                                      int sps, int dns, int k_split) {
//        for (int i = 0; i < this->num_layers - 1; i++) {
//            // SAND
//            GcnOpsOrder order = gemm_first;
//            if (this->weights[i]->ncols() >= this->weights[i]->nrows()) {
//                order = spmm_first;
//            }
//            gcn_forward_layer_slice_dgl_tiled<SM, DM1>(this->nadj,
//                                                       this->layer_acts[i],
//                                                       this->weights[i],
//                                                       this->biases[i],
//                                                       this->layer_acts[i + 1],
//                                                       order,
//                                                       spmm_variation,
//                                                       tile_size,
//                                                       refresh_and_print && (i == this->num_layers - 2),
//                                                       sps,
//                                                       dns,
//                                                       k_split);
//        }
//    }

    void clear() {
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->layer_acts.at(i + 1)->clear();
        }
    }

};

template<class SM, class DM1, class DM2>
class GCN_GEMM : public GNN<SM, DM1, DM2> {
    //GCN constructor
    //Values of constant matrices are shallow-copied!
    //Heap space is allocated for the output of each layer and the corresponding intermediate gradient.
    //TODO: If this is a problem perform optimizations.
public:
    GCN_GEMM(SM *nadj_,
             DM1 *input_emb_,
             DM2 *labels_,
             DM2 *train_masks_,
             DM2 *valid_masks_,
             DM2 *test_masks_,
             std::vector<DM1 *> weights_,
             std::vector<DM1 *> biases_,
             int num_layers_) {

        this->num_nodes = nadj_->nrows();
        this->num_layers = num_layers_;

        this->num_neurons.push_back(weights_[0]->nrows());
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->num_neurons.push_back(weights_[i]->ncols());
        }

        //Shallow copies
        this->nadj = nadj_;
        this->labels = labels_;
        this->train_masks = train_masks_;
        this->valid_masks = valid_masks_;
        this->test_masks = test_masks_;
        this->num_classes = this->num_neurons.back();

        this->individual_layer_times = new double *[this->num_layers - 1];

        //Deep copies
        //We are either initializing the weights to the initialization tensorflow values or the post-training ones.

        this->layer_acts.push_back(input_emb_);

        for (int i = 0; i < this->num_layers - 1; i++) {
            this->individual_layer_times[i] = new double[TIMES_TO_RECORD];
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                this->individual_layer_times[i][time_i] = 0;
            }
            this->weights.push_back(weights_[i]);
            this->biases.push_back(biases_[i]);

            //Allocating dense matrix objects for layer activations-not storing any values.
            DM1 *new_layer_act = new DM1;
            //For the time being we are storing activation matrices in the form Znum_neuronsxnum_nodes.
            //TODO: Change to transpose if better locality is exhibited.
            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type(), 0);
            this->layer_acts.push_back(new_layer_act);
        }
        this->degrees = new DM1;
        this->degrees->build(nadj_->nrows(), 1, weights_[0]->type(), 0);
        SpMV_ones(this->nadj, this->degrees);
        auto inverse_root_operator = inverse_root<typename DM1::vtype>;
        UEwD(this->degrees, this->degrees, inverse_root_operator);
        MMbroacast_row(this->layer_acts[0], this->degrees, this->layer_acts[0]);

    }

    void forward_pass(SpmmVariation spmm_variation,
                      int tile_size,
                      bool refresh_and_print) {
        for (int i = 0; i < this->num_layers - 1; i++) {
            // SAND
            // Do update or aggregate first
            GcnOpsOrder order = gemm_first;
            if (this->weights[i]->ncols() >= this->weights[i]->nrows()) {
                order = spmm_first;
            }
            if (i != 0) {
                MMbroacast_row(this->layer_acts[i], this->degrees, this->layer_acts[i]);
            }
            // Forward function
            gcn_forward_layer_gn2<SM, DM1>(this->nadj,
                                           this->degrees,
                                           this->layer_acts[i],
                                           this->weights[i],
                                           this->biases[i],
                                           this->layer_acts[i + 1],
                                           order,
                                           spmm_variation,
                                           tile_size,
                                           refresh_and_print && (i == this->num_layers - 2),
                                           i,
                                           this->individual_layer_times);
        }
        if (refresh_and_print) {
            double sums[TIMES_TO_RECORD] = {0};
            for (int i = 0; i < this->num_layers - 1; i++) {
                std::cout << "Times between layers " << i << " and " << i + 1 << ": ";
                for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                    sums[time_i] += this->individual_layer_times[i][time_i];
                    std::cout << this->individual_layer_times[i][time_i] << ",";
                }
                std::cout << std::endl;
            }
            std::cout << "*****" << std::endl;
            std::cout << "Summed times of operations: ";
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                std::cout << sums[time_i] << ",";
            }
            std::cout << std::endl;
        }
    }

    void clear() {
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->layer_acts.at(i + 1)->clear();
        }
    }

};

template<class SM, class DM1, class DM2>
class GCN_PRETILE : public GNN<SM, DM1, DM2> {
    std::vector<SM *> adj_vec;

public:
    GCN_PRETILE(std::vector<SM *> adj_vec_,
                DM1 *input_emb_,
                DM2 *labels_,
                DM2 *train_masks_,
                DM2 *valid_masks_,
                DM2 *test_masks_,
                std::vector<DM1 *> weights_,
                std::vector<DM1 *> biases_,
#ifdef GN_2
            DM1 *degrees_,
#endif
                int num_layers_) {

        this->num_nodes = input_emb_->nrows();
        this->num_layers = num_layers_;

        this->num_neurons.push_back(weights_[0]->nrows());
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->num_neurons.push_back(weights_[i]->ncols());
        }

        //Shallow copies
        this->adj_vec = adj_vec_;
        this->labels = labels_;
        this->train_masks = train_masks_;
        this->valid_masks = valid_masks_;
        this->test_masks = test_masks_;
        this->num_classes = this->num_neurons.back();

        this->layer_acts.push_back(input_emb_);

        this->individual_layer_times = new double *[this->num_layers - 1];

        for (int i = 0; i < this->num_layers - 1; i++) {

            this->individual_layer_times[i] = new double[TIMES_TO_RECORD];
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                this->individual_layer_times[i][time_i] = 0;
            }
            this->weights.push_back(weights_[i]);
            this->biases.push_back(biases_[i]);

            //Allocating dense matrix objects for layer activations-not storing any values.
            DM1 *new_layer_act = new DM1;
            //For the time being we are storing activation matrices in the form Znum_neuronsxnum_nodes.
            //TODO: Change to transpose if better locality is exhibited.
#ifdef A_ALLOC
            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type(), 0);
#else
            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type());
#endif
            this->layer_acts.push_back(new_layer_act);
        }
#ifdef GN_2
        this->degrees = degrees_;
        auto inverse_root_operator = inverse_root<typename DM1::vtype>;
        UEwD(this->degrees, this->degrees, inverse_root_operator);
        MMbroacast_row(this->layer_acts[0], this->degrees, this->layer_acts[0]);
#endif
    }

    void forward_pass_split_tiled(bool refresh_and_print, int sps
#if defined(ST_1) || defined(ST_2)
            , int dns
#endif
#ifdef LO_KK
            , int slice
#endif
    ) {
        for (int i = 0; i < this->num_layers - 1; i++) {
            // SAND
            GcnOpsOrder order = gemm_first;
            if (this->weights[i]->ncols() >= this->weights[i]->nrows()) {
                order = spmm_first;
            }

#ifdef GN_2
            if (i != 0) {
                MMbroacast_row(this->layer_acts[i], this->degrees, this->layer_acts[i]);
            }
#endif

            gcn_forward_layer_split_tiled<SM, DM1>(this->adj_vec,
#ifdef GN_2
                    this->degrees,
#endif
                                                   this->layer_acts[i],
                                                   this->weights[i],
                                                   this->biases[i],
                                                   this->layer_acts[i + 1],
                                                   order,
                                                   refresh_and_print && (i == this->num_layers - 2),
                                                   sps,
#if defined(ST_1) || defined(ST_2)
                    dns,
#endif
#ifdef LO_KK
                    slice,
#endif
                                                   i,
                                                   this->individual_layer_times);
        }
        if (refresh_and_print) {
            double sums[TIMES_TO_RECORD] = {0};
            for (int i = 0; i < this->num_layers - 1; i++) {
                std::cout << "Times between layers " << i << " and " << i + 1 << ": ";
                for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                    sums[time_i] += this->individual_layer_times[i][time_i];
                    std::cout << this->individual_layer_times[i][time_i] << ",";
                    this->individual_layer_times[i][time_i] = 0;
                }
                std::cout << std::endl;
            }
            std::cout << "*****" << std::endl;
            std::cout << "Summed times of operations: ";
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                std::cout << sums[time_i] << ",";
            }
            std::cout << std::endl;
        }
    }


    void forward_pass_split_tiled(bool refresh_and_print, std::vector<std::vector<GNOpTile<SM, DM1>>> &tile_infos) {
        for (int i = 0; i < this->num_layers - 1; i++) {
            // SAND
            GcnOpsOrder order = gemm_first;
            if (this->weights[i]->ncols() >= this->weights[i]->nrows()) {
                order = spmm_first;
            }
            gcn_forward_layer_split_tiled<SM, DM1>(this->adj_vec,
                                                   this->layer_acts[i],
                                                   this->weights[i],
                                                   this->biases[i],
                                                   this->layer_acts[i + 1],
                                                   order,
                                                   refresh_and_print && (i == this->num_layers - 2),
                                                   tile_infos,
                                                   i,
                                                   this->individual_layer_times);
        }
        if (refresh_and_print) {
            double sums[TIMES_TO_RECORD] = {0};
            for (int i = 0; i < this->num_layers - 1; i++) {
                std::cout << "Times between layers " << i << " and " << i + 1 << ": ";
                for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                    sums[time_i] += this->individual_layer_times[i][time_i];
                    std::cout << this->individual_layer_times[i][time_i] << ",";
                    this->individual_layer_times[i][time_i] = 0;
                }
                std::cout << std::endl;
            }
            std::cout << "*****" << std::endl;
            std::cout << "Summed times of operations: ";
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                std::cout << sums[time_i] << ",";
            }
            std::cout << std::endl;
        }
    }


    void forward_pass_split_tiled_seg(bool refresh_and_print, int sps, int seg_size) {
        for (int i = 0; i < this->num_layers - 1; i++) {
            // SAND
            GcnOpsOrder order = gemm_first;
            if (this->weights[i]->ncols() >= this->weights[i]->nrows()) {
                order = spmm_first;
            }
            gcn_forward_layer_split_segmented_tiled<SM, DM1>(this->adj_vec,
                                                             this->layer_acts[i],
                                                             this->weights[i],
                                                             this->biases[i],
                                                             this->layer_acts[i + 1],
                                                             order,
                                                             refresh_and_print && (i == this->num_layers - 2),
                                                             sps,
                                                             seg_size,
                                                             i,
                                                             this->individual_layer_times);
        }
        if (refresh_and_print) {
            double sums[TIMES_TO_RECORD] = {0};
            for (int i = 0; i < this->num_layers - 1; i++) {
                std::cout << "Times between layers " << i << " and " << i + 1 << ": ";
                for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                    sums[time_i] += this->individual_layer_times[i][time_i];
                    std::cout << this->individual_layer_times[i][time_i] << ",";
                    this->individual_layer_times[i][time_i] = 0;
                }
                std::cout << std::endl;
            }
            std::cout << "*****" << std::endl;
            std::cout << "Summed times of operations: ";
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                std::cout << sums[time_i] << ",";
            }
            std::cout << std::endl;
        }
    }

    void forward_pass_slice_split_tiled(bool refresh_and_print, int sps, int slice_size) {
        for (int i = 0; i < this->num_layers - 1; i++) {
            // SAND
            GcnOpsOrder order = gemm_first;
            if (this->weights[i]->ncols() >= this->weights[i]->nrows()) {
                order = spmm_first;
            }
            gcn_forward_layer_slice_split_tiled<SM, DM1>(this->adj_vec,
                                                         this->layer_acts[i],
                                                         this->weights[i],
                                                         this->biases[i],
                                                         this->layer_acts[i + 1],
                                                         order,
                                                         refresh_and_print && (i == this->num_layers - 2),
                                                         sps,
                                                         slice_size,
                                                         i,
                                                         this->individual_layer_times);
        }
        if (refresh_and_print) {
            double sums[TIMES_TO_RECORD] = {0};
            for (int i = 0; i < this->num_layers - 1; i++) {
                std::cout << "Times between layers " << i << " and " << i + 1 << ": ";
                for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                    sums[time_i] += this->individual_layer_times[i][time_i];
                    std::cout << this->individual_layer_times[i][time_i] << ",";
                    this->individual_layer_times[i][time_i] = 0;
                }
                std::cout << std::endl;
            }
            std::cout << "*****" << std::endl;
            std::cout << "Summed times of operations: ";
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                std::cout << sums[time_i] << ",";
            }
            std::cout << std::endl;
        }
    }

    void clear() {
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->layer_acts.at(i + 1)->clear();
        }
    }
};

template<class SM, class DM1, class DM2>
class GCN_PRETILE_GEMM : public GNN<SM, DM1, DM2> {
    std::vector<SM *> adj_vec;

public:
    GCN_PRETILE_GEMM(std::vector<SM *> adj_vec_,
                     DM1 *input_emb_,
                     DM2 *labels_,
                     DM2 *train_masks_,
                     DM2 *valid_masks_,
                     DM2 *test_masks_,
                     std::vector<DM1 *> weights_,
                     std::vector<DM1 *> biases_,
                     DM1 *degrees_,
                     int num_layers_) {

        this->num_nodes = input_emb_->nrows();
        this->num_layers = num_layers_;

        this->num_neurons.push_back(weights_[0]->nrows());
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->num_neurons.push_back(weights_[i]->ncols());
        }

        //Shallow copies
        this->adj_vec = adj_vec_;
        this->labels = labels_;
        this->train_masks = train_masks_;
        this->valid_masks = valid_masks_;
        this->test_masks = test_masks_;
        this->num_classes = this->num_neurons.back();

        this->layer_acts.push_back(input_emb_);

        this->individual_layer_times = new double *[this->num_layers - 1];

        for (int i = 0; i < this->num_layers - 1; i++) {

            this->individual_layer_times[i] = new double[TIMES_TO_RECORD];
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                this->individual_layer_times[i][time_i] = 0;
            }
            this->weights.push_back(weights_[i]);
            this->biases.push_back(biases_[i]);

            //Allocating dense matrix objects for layer activations-not storing any values.
            DM1 *new_layer_act = new DM1;
            //For the time being we are storing activation matrices in the form Znum_neuronsxnum_nodes.
            //TODO: Change to transpose if better locality is exhibited.
            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type(), 0);
            this->layer_acts.push_back(new_layer_act);
        }
        this->degrees = degrees_;
        auto inverse_root_operator = inverse_root<typename DM1::vtype>;
        UEwD(this->degrees, this->degrees, inverse_root_operator);
        MMbroacast_row(this->layer_acts[0], this->degrees, this->layer_acts[0]);
    }

    void forward_pass_split_tiled(bool refresh_and_print, int sps
#if defined(ST_1) || defined(ST_2)
            , int dns
#endif
#ifdef LO_KK
            , int slice
#endif
    ) {
        for (int i = 0; i < this->num_layers - 1; i++) {
            // SAND
            GcnOpsOrder order = gemm_first;
            if (this->weights[i]->ncols() >= this->weights[i]->nrows()) {
                order = spmm_first;
            }

            if (i != 0) {
                MMbroacast_row(this->layer_acts[i], this->degrees, this->layer_acts[i]);
            }

            gcn_forward_layer_split_tiled_gn2<SM, DM1>(this->adj_vec,
                                                       this->degrees,
                                                       this->layer_acts[i],
                                                       this->weights[i],
                                                       this->biases[i],
                                                       this->layer_acts[i + 1],
                                                       order,
                                                       refresh_and_print && (i == this->num_layers - 2),
                                                       sps,
#if defined(ST_1) || defined(ST_2)
                    dns,
#endif
#ifdef LO_KK
                    slice,
#endif
                                                       i,
                                                       this->individual_layer_times);
        }
        if (refresh_and_print) {
            double sums[TIMES_TO_RECORD] = {0};
            for (int i = 0; i < this->num_layers - 1; i++) {
                std::cout << "Times between layers " << i << " and " << i + 1 << ": ";
                for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                    sums[time_i] += this->individual_layer_times[i][time_i];
                    std::cout << this->individual_layer_times[i][time_i] << ",";
                    this->individual_layer_times[i][time_i] = 0;
                }
                std::cout << std::endl;
            }
            std::cout << "*****" << std::endl;
            std::cout << "Summed times of operations: ";
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                std::cout << sums[time_i] << ",";
            }
            std::cout << std::endl;
        }
    }

    void clear() {
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->layer_acts.at(i + 1)->clear();
        }
    }
};

// TODO create a GCN class for ASpT

//template<class SM, class DM1, class DM2>
//class GCN_GEMM : public GNN<SM, DM1, DM2> {
//
//    //GCN constructor
//    //Values of constant matrices are shallow-copied!
//    //Heap space is allocated for the output of each layer and the corresponding intermediate gradient.
//    //TODO: If this is a problem perform optimizations.
//public:
//    GCN_GEMM(SM *nadj_, DM1 *input_emb_, DM2 *labels_, DM2 *train_masks_,
//             DM2 *valid_masks_, DM2 *test_masks_, std::vector<DM1 *> weights_, std::vector<DM1 *> biases_,
//             int num_layers_) {
//
//        this->num_nodes = nadj_->nrows();
//        this->num_layers = num_layers_;
//
//        this->num_neurons.push_back(weights_[0]->nrows());
//        for (int i = 0; i < this->num_layers - 1; i++) {
//            this->num_neurons.push_back(weights_[i]->ncols());
//        }
//
//        //Shallow copies
//        this->nadj = nadj_;
//        this->labels = labels_;
//        this->train_masks = train_masks_;
//        this->valid_masks = valid_masks_;
//        this->test_masks = test_masks_;
//        this->num_classes = this->num_neurons.back();
//
//        //Deep copies
//        //We are either initializing the weights to the initialization tensorflow values or the post-training ones.
//
//        this->layer_acts.push_back(input_emb_);
//
//        for (int i = 0; i < this->num_layers - 1; i++) {
//            //std::cout<<"Generated intermediate matrices for layer "<<i + 1<<std::endl;
//            //weights[0] corresponds to weights between layers 0 and 1.
//
//            DM1 *new_weights = new DM1;
//            new_weights->build(weights_[i]->nrows(), weights_[i]->ncols(), weights_[i]->vals_ptr(),
//                               weights_[i]->type());
//            this->weights.push_back(new_weights);
//            DM1 *new_biases = new DM1;
//            new_biases->build(biases_[i]->nrows(), biases_[i]->ncols(), biases_[i]->vals_ptr(), biases_[i]->type());
//            this->biases.push_back(new_biases);
//
//            //Allocating dense matrix objects for layer activations-not storing any values.
//            DM1 *new_layer_act = new DM1;
//            //For the time being we are storing activation matrices in the form Znum_neuronsxnum_nodes.
//            //TODO: Change to transpose if better locality is exhibited.
//            //std::cout<<"H"<<std::endl;
//            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type());
//            this->layer_acts.push_back(new_layer_act);
//        }
//
//        // SAND
//        this->degrees = new DM1;
//        this->degrees->build(nadj_->nrows(), 1, weights_[0]->type());
//        SpMV_ones(this->nadj, this->degrees);
//        auto inverse_root_operator = inverse_root<typename DM1::vtype>;
//        UEwD(this->degrees, this->degrees, inverse_root_operator);
//        MMbroacast_row(this->layer_acts[0], this->degrees, this->layer_acts[0]);
//
//    }
//
//    void forward_pass(SpmmVariation spmm_variation, int tile_size, bool refresh_and_print) {
//        for (int i = 0; i < this->num_layers - 1; i++) {
//            // SAND
//            GcnOpsOrder order = gemm_first;
//            if (this->weights[i]->ncols() >= this->weights[i]->nrows()) {
//                order = spmm_first;
//            }
//            if (i != 0) {
//                MMbroacast_row(this->layer_acts[i], this->degrees, this->layer_acts[i]);
//            }
//            gcn_gemm_forward_layer<SM, DM1>(this->nadj,
//                                            this->degrees,
//                                            this->layer_acts[i],
//                                            this->weights[i],
//                                            this->biases[i],
//                                            this->layer_acts[i + 1],
//                                            order,
//                                            spmm_variation,
//                                            tile_size,
//                                            refresh_and_print && (i == this->num_layers - 2));
//        }
//    }
//};


template<class SM, class DM1, class DM2>
class GATB : public GNN<SM, DM1, DM2> {
public:
    std::vector<DM1 *> a_l_weights_;
    std::vector<DM1 *> a_r_weights_;

    std::vector<DM1 *> wh_l_s;
    std::vector<DM1 *> wh_r_s;

    // TODO you only need one of these

    GATB(SM *adj_,
         DM1 *input_emb_,
         DM2 *labels_,
         DM2 *train_masks_,
         DM2 *valid_masks_,
         DM2 *test_masks_,
         std::vector<DM1 *> weights_,
         std::vector<DM1 *> biases_,
         std::vector<DM1 *> a_ls,
         std::vector<DM1 *> a_rs,
         int num_layers_) {

        this->num_nodes = adj_->nrows();
        this->num_layers = num_layers_;

        this->num_neurons.push_back(weights_[0]->nrows());
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->num_neurons.push_back(weights_[i]->ncols());
        }

        //Shallow copies
        this->nadj = adj_;
        this->labels = labels_;
        this->train_masks = train_masks_;
        this->valid_masks = valid_masks_;
        this->test_masks = test_masks_;
        this->num_classes = this->num_neurons.back();

        this->individual_layer_times = new double *[this->num_layers - 1];

        //Deep copies
        //We are either initializing the weights to the initialization tensorflow values or the post-training ones.

        this->layer_acts.push_back(input_emb_);

        for (int i = 0; i < this->num_layers - 1; i++) {
            //std::cout << "Generated intermediate matrices for layer " << i + 1 << std::endl;
            //weights[0] corresponds to weights between layers 0 and 1.
            this->individual_layer_times[i] = new double[TIMES_TO_RECORD];
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                this->individual_layer_times[i][time_i] = 0;
            }

            this->weights.push_back(weights_[i]);
            this->biases.push_back(biases_[i]);
            this->a_l_weights_.push_back(a_ls[i]);
            this->a_r_weights_.push_back(a_rs[i]);

            //Allocating dense matrix objects for layer activations-not storing any values.
            DM1 *new_layer_act = new DM1;
            //For the time being we are storing activation matrices in the form Znum_neuronsxnum_nodes.
            //TODO: Change to transpose if better locality is exhibited.
#ifdef A_ALLOC
            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type(), 0);
#else
            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type());
#endif
            this->layer_acts.push_back(new_layer_act);

            DM1 *new_whr = new DM1;
#ifdef A_ALLOC
            new_whr->build(this->num_nodes, 1, input_emb_->type(), 0);
#else
            new_whr->build(this->num_nodes, 1, input_emb_->type());
#endif
            this->wh_r_s.push_back(new_whr);

            DM1 *new_whl = new DM1;
#ifdef A_ALLOC
            new_whl->build(this->num_nodes, 1, input_emb_->type(), 0);
#else
            new_whl->build(this->num_nodes, 1, input_emb_->type());
#endif
            this->wh_l_s.push_back(new_whl);
        }

    }

    void forward_pass(bool recomp, bool refresh_and_print) {
        for (int i = 0; i < this->num_layers - 1; i++) {
            gat_forward_layer_wbias<SM, DM1>(this->nadj,
                                             this->layer_acts[i],
                                             this->weights[i],
                                             this->biases[i],
                                             this->a_l_weights_[i],
                                             this->a_r_weights_[i],
                                             this->layer_acts[i + 1],
                                             this->wh_l_s[i],
                                             this->wh_r_s[i],
                                             recomp,
                                             refresh_and_print && (i == this->num_layers - 2),
                                             i,
                                             this->individual_layer_times);
        }
        if (refresh_and_print) {
            double sums[TIMES_TO_RECORD] = {0};
            for (int i = 0; i < this->num_layers - 1; i++) {
                std::cout << "Times between layers " << i << " and " << i + 1 << ": ";
                for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                    sums[time_i] += this->individual_layer_times[i][time_i];
                    std::cout << this->individual_layer_times[i][time_i] << ",";
                    this->individual_layer_times[i][time_i] = 0;
                }
                std::cout << std::endl;
            }
            std::cout << "*****" << std::endl;
            std::cout << "Summed times of operations: ";
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                std::cout << sums[time_i] << ",";
            }
            std::cout << std::endl;
        }
    }

    void clear() {
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->layer_acts.at(i + 1)->clear();
            this->wh_r_s.at(i)->clear();
            this->wh_l_s.at(i)->clear();
        }
    }
};

template<class SM, class DM1, class DM2>
class GAT : public GNN<SM, DM1, DM2> {
public:
    std::vector<DM1 *> a_l_weights_;
    std::vector<DM1 *> a_r_weights_;

    std::vector<DM1 *> wh_l_s;
    std::vector<DM1 *> wh_r_s;

    // TODO you only need one of these

    GAT(SM *adj_,
        DM1 *input_emb_,
        DM2 *labels_,
        DM2 *train_masks_,
        DM2 *valid_masks_,
        DM2 *test_masks_,
        std::vector<DM1 *> weights_,
        std::vector<DM1 *> a_ls,
        std::vector<DM1 *> a_rs,
        int num_layers_) {

        this->num_nodes = adj_->nrows();
        this->num_layers = num_layers_;

        this->num_neurons.push_back(weights_[0]->nrows());
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->num_neurons.push_back(weights_[i]->ncols());
        }

        //Shallow copies
        this->nadj = adj_;
        this->labels = labels_;
        this->train_masks = train_masks_;
        this->valid_masks = valid_masks_;
        this->test_masks = test_masks_;
        this->num_classes = this->num_neurons.back();

        this->individual_layer_times = new double *[this->num_layers - 1];

        //Deep copies
        //We are either initializing the weights to the initialization tensorflow values or the post-training ones.

        this->layer_acts.push_back(input_emb_);

        for (int i = 0; i < this->num_layers - 1; i++) {
            //std::cout << "Generated intermediate matrices for layer " << i + 1 << std::endl;
            //weights[0] corresponds to weights between layers 0 and 1.
            this->individual_layer_times[i] = new double[TIMES_TO_RECORD];
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                this->individual_layer_times[i][time_i] = 0;
            }

            this->weights.push_back(weights_[i]);
            this->a_l_weights_.push_back(a_ls[i]);
            this->a_r_weights_.push_back(a_rs[i]);

            //Allocating dense matrix objects for layer activations-not storing any values.
            DM1 *new_layer_act = new DM1;
            //For the time being we are storing activation matrices in the form Znum_neuronsxnum_nodes.
            //TODO: Change to transpose if better locality is exhibited.
#ifdef A_ALLOC
            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type(), 0);
#else
            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type());
#endif
            this->layer_acts.push_back(new_layer_act);

            DM1 *new_whr = new DM1;
#ifdef A_ALLOC
            new_whr->build(this->num_nodes, 1, input_emb_->type(), 0);
#else
            new_whr->build(this->num_nodes, 1, input_emb_->type());
#endif
            this->wh_r_s.push_back(new_whr);

            DM1 *new_whl = new DM1;
#ifdef A_ALLOC
            new_whl->build(this->num_nodes, 1, input_emb_->type(), 0);
#else
            new_whl->build(this->num_nodes, 1, input_emb_->type());
#endif
            this->wh_l_s.push_back(new_whl);
        }

    }

    void forward_pass(bool recomp, bool refresh_and_print) {
        for (int i = 0; i < this->num_layers - 1; i++) {
            gat_forward_layer<SM, DM1>(this->nadj,
                                       this->layer_acts[i],
                                       this->weights[i],
                                       this->a_l_weights_[i],
                                       this->a_r_weights_[i],
                                       this->layer_acts[i + 1],
                                       this->wh_l_s[i],
                                       this->wh_r_s[i],
                                       recomp,
                                       refresh_and_print && (i == this->num_layers - 2),
                                       i,
                                       this->individual_layer_times);
        }
        if (refresh_and_print) {
            double sums[TIMES_TO_RECORD] = {0};
            for (int i = 0; i < this->num_layers - 1; i++) {
                std::cout << "Times between layers " << i << " and " << i + 1 << ": ";
                for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                    sums[time_i] += this->individual_layer_times[i][time_i];
                    std::cout << this->individual_layer_times[i][time_i] << ",";
                    this->individual_layer_times[i][time_i] = 0;
                }
                std::cout << std::endl;
            }
            std::cout << "*****" << std::endl;
            std::cout << "Summed times of operations: ";
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                std::cout << sums[time_i] << ",";
            }
            std::cout << std::endl;
        }
    }

    void clear() {
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->layer_acts.at(i + 1)->clear();
            this->wh_r_s.at(i)->clear();
            this->wh_l_s.at(i)->clear();
        }
    }
};

template<class SM, class DM1, class DM2>
class GAT_PRETILE : public GNN<SM, DM1, DM2> {
public:
    std::vector<DM1 *> a_l_weights_;
    std::vector<DM1 *> a_r_weights_;

    std::vector<DM1 *> wh_l_s;
    std::vector<DM1 *> wh_r_s;

    std::vector<SM *> adj_vec;

    GAT_PRETILE(std::vector<SM *> adj_vec_,
                DM1 *input_emb_,
                DM2 *labels_,
                DM2 *train_masks_,
                DM2 *valid_masks_,
                DM2 *test_masks_,
                std::vector<DM1 *> weights_,
                std::vector<DM1 *> a_ls,
                std::vector<DM1 *> a_rs,
                int num_layers_) {

        this->num_nodes = input_emb_->nrows();
        this->num_layers = num_layers_;

        this->num_neurons.push_back(weights_[0]->nrows());
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->num_neurons.push_back(weights_[i]->ncols());
        }

        //Shallow copies
        this->adj_vec = adj_vec_;
        this->labels = labels_;
        this->train_masks = train_masks_;
        this->valid_masks = valid_masks_;
        this->test_masks = test_masks_;
        this->num_classes = this->num_neurons.back();

        this->individual_layer_times = new double *[this->num_layers - 1];

        //Deep copies
        //We are either initializing the weights to the initialization tensorflow values or the post-training ones.

        this->layer_acts.push_back(input_emb_);

        for (int i = 0; i < this->num_layers - 1; i++) {
            //std::cout << "Generated intermediate matrices for layer " << i + 1 << std::endl;
            //weights[0] corresponds to weights between layers 0 and 1.
            this->individual_layer_times[i] = new double[TIMES_TO_RECORD];
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                this->individual_layer_times[i][time_i] = 0;
            }

            this->weights.push_back(weights_[i]);
            this->a_l_weights_.push_back(a_ls[i]);
            this->a_r_weights_.push_back(a_rs[i]);

            //Allocating dense matrix objects for layer activations-not storing any values.
            DM1 *new_layer_act = new DM1;
            //For the time being we are storing activation matrices in the form Znum_neuronsxnum_nodes.
            //TODO: Change to transpose if better locality is exhibited.
#ifdef A_ALLOC
            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type(), 0);
#else
            new_layer_act->build(this->num_nodes, this->num_neurons[i + 1], input_emb_->type());
#endif
            this->layer_acts.push_back(new_layer_act);

            DM1 *new_whr = new DM1;
#ifdef A_ALLOC
            new_whr->build(this->num_nodes, 1, input_emb_->type(), 0);
#else
            new_whr->build(this->num_nodes, 1, input_emb_->type());
#endif
            this->wh_r_s.push_back(new_whr);

            DM1 *new_whl = new DM1;
#ifdef A_ALLOC
            new_whl->build(this->num_nodes, 1, input_emb_->type(), 0);
#else
            new_whl->build(this->num_nodes, 1, input_emb_->type());
#endif
            this->wh_l_s.push_back(new_whl);
        }

    }

    void forward_pass(bool refresh_and_print, int sps, bool recomp
#if defined(ST_1) || defined(ST_2)
            , int dns
#endif
#ifdef LO_KK
            , int slice
#endif
    ) {
        for (int i = 0; i < this->num_layers - 1; i++) {
            gat_forward_layer_split_tiled<SM, DM1>(this->adj_vec,
                                                   this->layer_acts[i],
                                                   this->weights[i],
                                                   this->a_l_weights_[i],
                                                   this->a_r_weights_[i],
                                                   this->layer_acts[i + 1],
                                                   this->wh_l_s[i],
                                                   this->wh_r_s[i],
                                                   recomp,
                                                   refresh_and_print && (i == this->num_layers - 2),
                                                   sps,
#if defined(ST_1) || defined(ST_2)
                    dns,
#endif
#ifdef LO_KK
                    slice,
#endif
                                                   i,
                                                   this->individual_layer_times);
        }
        if (refresh_and_print) {
            double sums[TIMES_TO_RECORD] = {0};
            for (int i = 0; i < this->num_layers - 1; i++) {
                std::cout << "Times between layers " << i << " and " << i + 1 << ": ";
                for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                    sums[time_i] += this->individual_layer_times[i][time_i];
                    std::cout << this->individual_layer_times[i][time_i] << ",";
                    this->individual_layer_times[i][time_i] = 0;
                }
                std::cout << std::endl;
            }
            std::cout << "*****" << std::endl;
            std::cout << "Summed times of operations: ";
            for (int time_i = 0; time_i < TIMES_TO_RECORD; time_i++) {
                std::cout << sums[time_i] << ",";
            }
            std::cout << std::endl;
        }
    }

    void clear() {
        for (int i = 0; i < this->num_layers - 1; i++) {
            this->layer_acts.at(i + 1)->clear();
            this->wh_r_s.at(i)->clear();
            this->wh_l_s.at(i)->clear();
        }
    }
};
