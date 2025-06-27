#ifndef FRONTEND_METADATA
#define FRONTEND_METADATA

#include <vector>
#include <map>
#include <string>
using namespace std;

typedef enum {
    GET_DEGREES,
    GET_NORMALIZATION,
    MULT_NORM_RES,
    MESSAGE_PASSING_AGGREGATE,
    FEED_FORWARD_NN,
    ADD_TWO_FFN,
    NON_LINEARITY,
    ATTEN_L,
    ATTEN_R,
    ATTN,
    LEAKY_RELU_OP,
    SAGE_OPS,
    SOFTMAX_OP,
    MULT_SCALAR_FEATS,
    ADD_SCALAR_AGGR
} LayerOpType;

typedef enum {
    UNDIRECTED,
    UNWEIGHTED,
    SPARSE,
    SAMP,
    FEAT_SIZE,
    LABEL_SIZE
} GraphTransformType;

typedef enum {
    COARSE,
    SAMP_DYN_CPT,
    SAMP_CPT
} ComputeTransformType;

typedef enum {
    COL_TILE
} DataTransformType;

class ModelConfig {
    public:
        string dataset_name;
        int iterations;
        vector<int> output_input_classes; // output of first layer fed into second layer
        int num_layers;
        int validation_step;
        float normalization_value;
        vector<LayerOpType> layer_operations;
        vector<bool> nonln_present;
        map<GraphTransformType, float> graph_transformations;
        vector<pair<ComputeTransformType, float>> compute_transformations;
        vector<pair<DataTransformType, float>> data_transformations;

        void addGraphTransformation(GraphTransformType t, float param){
            graph_transformations[t] = param;
        }
        void addComputeTransformation(ComputeTransformType t, float param){
            compute_transformations.push_back({t,  param});
        }
        void addDataTransformation(DataTransformType t, float param){
            data_transformations.push_back({t,  param});
        }
        ModelConfig(){
            dataset_name = "\0";
            iterations = 0;
            num_layers = 0;
            normalization_value = -1;
            normalization_value = 0;
            output_input_classes.clear();
            layer_operations.clear();
            graph_transformations.clear();
            graph_transformations[UNDIRECTED] = true;
            graph_transformations[UNWEIGHTED] = true;
            graph_transformations[FEAT_SIZE] = -2;
            graph_transformations[LABEL_SIZE] = -3;
            compute_transformations.clear();
            data_transformations.clear();
        }
        string to_string(LayerOpType t) {
            switch (t) {
                case GET_DEGREES: return "GET_DEGREES";
                case GET_NORMALIZATION: return "GET_NORMALIZATION";
                case MULT_NORM_RES: return "MULT_NORM_RES";
                case MESSAGE_PASSING_AGGREGATE: return "MESSAGE_PASSING_AGGREGATE";
                case FEED_FORWARD_NN: return "FEED_FORWARD_NN";
                case NON_LINEARITY: return "NON_LINEARITY";
                case ATTEN_L: return "ATTEN_L";
                case ATTEN_R: return "ATTEN_R";
                case LEAKY_RELU_OP: return "LEAKY_RELU_OP";
                case SOFTMAX_OP: return "SOFTMAX_OP";
                case MULT_SCALAR_FEATS: return "MULT_SCALAR_FEATS";
                case ADD_SCALAR_AGGR: return "ADD_SCALAR_AGGR";
                case SAGE_OPS: return "SAGE_OPS";
                case ADD_TWO_FFN: return "ADD_TWO_FFN";
                default: return "UNKNOWN_LAYER_OP";
            }
        }

        string to_string(GraphTransformType t) {
            switch (t) {
                case UNDIRECTED: return "UNDIRECTED";
                case UNWEIGHTED: return "UNWEIGHTED";
                case FEAT_SIZE: return "FEAT_SIZE";
                case LABEL_SIZE: return "LABEL_SIZE";
                default: return "UNKNOWN_GRAPH_TRANSFORM";
            }
        }

        string to_string(ComputeTransformType t) {
            switch (t) {
                case COARSE: return "COARSE";
                default: return "UNKNOWN_COMPUTE_TRANSFORM";
            }
        }

        string to_string(DataTransformType t) {
            switch (t) {
                case COL_TILE: return "COL_TILE";
                default: return "UNKNOWN_DATA_TRANSFORM";
            }
        }
        string to_string() {
            string a = "Model Configuration:\n";
            a += "Dataset Name: " + dataset_name + "\n";
            a += "Iterations: " + std::to_string(iterations) + "\n";
            // a += "Output Input Classes: " + std::to_string(output_input_classes) + "\n";
            a += "Number of Layers: " + std::to_string(num_layers) + "\n";
            a += "NonLn Vector: ";
            for (int b : nonln_present) a += std::to_string(b) + ' ';
            a += '\n';
            a += "Output Input Classes Vector: ";
            for (int b : output_input_classes) a += std::to_string(b) + ' ';
            a += '\n';
            a += "Layer Operations:\n";
            for (const auto& op : layer_operations) {
                a += "  - " + to_string(op) + "\n";
            }
            a += "Graph Transformations:\n";
            for (const auto& pair : graph_transformations) {
                a += "  - " + to_string(pair.first) + " (param: " + std::to_string(pair.second) + ")\n";
            }
            a += "Compute Transformations:\n";
            for (const auto& pair : compute_transformations) {
                a += "  - " + to_string(pair.first) + " (param: " + std::to_string(pair.second) + ")\n";
            }
            a += "Data Transformations:\n";
            for (const auto& pair : data_transformations) {
                a += "  - " + to_string(pair.first) + " (param: " + std::to_string(pair.second) + ")\n";
            }
            return a;
        }

};

#endif // FRONTEND_METADATA