//
// Created by damitha on 4/1/24.
//

#ifndef GNN_ACCELERATION_LANGUAGE_COMPUTE_H
#define GNN_ACCELERATION_LANGUAGE_COMPUTE_H

#include <any>
//#include <memory>
#include "data.h"

enum LossFunction {
  CROSS_ENTROPY,
};

enum NNOptimizer {
  ADAM,
};

enum NonLnrOp {
    ReLU_OP,
    LeakyReLU_OP
};

enum OpType {
    POINTWISE, // Pointwise update of data
    AGGREGATE_EDGE, // Edge aggregation operation (SDDMM-based)
    AGGREGATE_NODE, // Node aggregation operation (SpMM-based)
    UPDATE_EDGE, // Edge update operation (Edge softmax)
    UPDATE_NODE, // Node update operation (NN, Non-linear op)
};

enum ComputeOp {
    LOAD_OP,
    DEGREES_OP,
    POWER_OP,
    APPLY_EDGES_OP, // SDDMM
    AGGREGATE_MUL_SUM_OP, // SpMM
    AGGREGATE_EDGE_SUM_OP,
    AGGREGATE_EDGE_MUL_OP,
    AGGREGATE_MUL_SUM_DIRECT, // No autograd
    FFN_OP, // Can also be UPDATE
    FFN_OP_EDGE,
    FFN_OP_SELF,
    BIAS_OP,
    NON_LNR_OP_RELU,
    NON_LNR_OP_LOG_SOFTMAX,
    NON_LNR_OP_SOFTMAX,
    NON_LNR_OP_LEAKY_RELU,
    ROW_BROADCAST_OP,
    SCALAR_ADD_EPS_MULTIPLY_OP,
    ADD_OP, // Tensor add
    MUL_OP, // Tensor multiply
    // Control statements
    IF_CONTROL,
    TRAIN_CONTROL,
    // Transformation
    TRANSFORM_OP,
    // Data creation ops
    ONES_OP,
    EPSILON_OP,
};

enum CompOptimization {
    COARSE_COPT, // Thread coarsening
    SAMPLE_COPT, // Sample an aggregation kernel
};

// TODO - Add node / edge aggregation types.

class CIRNode{
public:
    CIRNode() {};
    virtual ~CIRNode() {};
};

class ComputeNode : virtual public CIRNode {
private:
  	OpType opType;
    ComputeOp op;

    std::vector<std::string> params;

    std::vector<DataNode*> inputData;
    std::vector<DataNode*> outputData;
    std::vector<std::pair<CompOptimization, float>> opts;

    std::string kernelName;

    // Points in the program are commented out for now
//    // Program start point
//    int point;
public:
    ComputeNode(OpType opType, ComputeOp op) {
        this->op = op;
        this->opType = opType;
    }

    // Op and the type shouldn't change

    // Op -- Also used to identify between the
    ComputeOp getOp() { return this->op;}
    OpType getOpType() { return this->opType;}


    // Get opts
  	std::vector<std::pair<CompOptimization, float>>* getOpts() {
    	return &opts;
  	};
  	// Add opts
  	void addOpt(CompOptimization opt, float param) {
    	opts.push_back(std::make_pair(opt, param));
  	};
    int getNumOpts() { return (int)opts.size(); };
    std::pair<CompOptimization, float>* getOpt(int idx)
    {
        return &opts.at(idx);
    }

    // Params can only add new (No need to remove any)
    void addParam(std::string new_param) { this->params.push_back(new_param); }
    // At Nihility's(IX) end
    std::string getParam(int ix) { return this->params.at(ix); }
    void setParam(int ix, std::string new_val) { this->params.at(ix) = new_val; }

    // Data items should be constant once added (No need to remove elements)
    //  TODO it can change if any data transformations are done on it.
    void addInputData(DataNode *new_input) { this->inputData.push_back(new_input); }
    void addOutputData(DataNode *new_output) { this->outputData.push_back(new_output); }

    long getNumInputs()
    {
       return this->inputData.size();
    }
    DataNode* getInput(int ix)
    {
        return this->inputData[ix];
    }
    void setInputDataNode(int ix, DataNode* new_input)
    {
        this->inputData[ix] = new_input;
    }
    long getNumOutputs()
    {
        return this->outputData.size();
    }
    DataNode* getOutput(int ix)
    {
        return this->outputData[ix];
    }
    void setOutputDataNode(int ix, DataNode* new_input)
    {
        this->outputData[ix] = new_input;
    }

    void setKernelName(std::string name) { this->kernelName = name; }
    std::string getKernelName() { return this->kernelName; }
};

class ForwardNode : public ComputeNode {
  private:
    ComputeNode* backward;
  public:
    ForwardNode(OpType opType, ComputeOp op): ComputeNode(opType, op) {};

    // Backward getter and setters
    ComputeNode* getBackward() { return this->backward; }
    void setBackward(ComputeNode* new_backward) { this->backward = new_backward; }
};

class TrainingLoopNode : public CIRNode  {
private:
    int numIter;
    int stepValid;
    int stepTest;
    float learningRate;
    LossFunction lossFunc;
    NNOptimizer optimizer;
    // The steps in the program validation happens. If not specified the numIter.
    // TODO need an easy way to find a node by an ID and then return + remove it.
    //  temp solution - Remove everything and then add everything back
    std::vector<ForwardNode *> loop;
public:
    TrainingLoopNode(int numIter, LossFunction lossFunc = CROSS_ENTROPY, NNOptimizer optimizer = ADAM, int stepValid = 0, int stepTest = 1, float learningRate = 0.01){
        this->numIter = numIter;
        this->lossFunc = lossFunc;
        this->optimizer = optimizer;
        this->stepValid = stepValid;
        this->stepTest = stepTest;
        this->learningRate = learningRate;
    }

    int getIter() { return this->numIter; }
    int getValidStep() { return this->stepValid; }
    int getTestStep() { return this->stepTest; }
    float getLearningRate() { return this->learningRate; }
    LossFunction getLossFunc() { return this->lossFunc; }
    NNOptimizer getOptimizer() { return this->optimizer; }

    void clearLoopNodes() { this->loop.clear(); }
    void addLoopNode(ForwardNode *newNode) { this->loop.push_back(newNode); }
    int getLoopNodeNum() { return this->loop.size(); }

    std::vector<ForwardNode *> *getLoopNodes() { return &loop; }
    ForwardNode *getNode(int i) { return this->loop.at(i); }
    void eraseFirstNLoopNodes(int n, int start = 0)
    {
        for (int i_n = start; i_n < start + n; i_n++)
        {
            this->loop.erase(this->loop.begin() + start);
        }
    }
    void insertToLoopNodes(int i, ForwardNode *newNode)
    {
        this->loop.insert(this->loop.begin() + i, newNode);
    }
    void swapNodes(int i, int j) { std::swap(this->loop[i], this->loop[j]); }
};

//class TrainingLoopNode : public ComputeNode {
//private:
//    int numIter;
//    int stepValid; // The steps in the program validation happens. If not specified the numIter.
//    // TODO need an easy way to find a node by an ID and then return + remove it.
//    //  temp solution - Remove everything and then add everything back
//    std::vector<ComputeNode *> loop;
//public:
//    TrainingLoopNode(int point, int numIter) : ComputeNode(TRAIN_CONTROL, point) {
//        this->numIter = numIter;
//        this->stepValid = numIter;
//    }
//
//    TrainingLoopNode(int point, int numIter, int stepValid) : ComputeNode(TRAIN_CONTROL, point) {
//        this->numIter = numIter;
//        this->stepValid = stepValid;
//    }
//
//    int getIter() { return this->numIter; }
//
//    int getValidStep() { return this->stepValid; }
//
//    void clearLoopNodes() { this->loop.clear(); }
//
//    void addLoopNode(ComputeNode *newNode) { this->loop.push_back(newNode); }
//
//    int getLoopNodeNum() { return this->loop.size(); }
//
//    std::vector<ComputeNode *> *getLoopNodes() { return &loop; }
//
//    ComputeNode *getNode(int i) { return this->loop.at(i); }
//};

//class TransformOpNode : public ComputeNode {
//private:
//    TransformEdge *transformData;
//public:
//    TransformOpNode(int point, TransformEdge *trEdge) : ComputeNode(TRANSFORM_OP, point) {
//        this->transformData = trEdge;
//    }
//
//    TransformEdge *getTransformations() { return this->transformData; }
//};
//
//class PointCounter {
//private:
//    int point;
//public:
//    PointCounter() { this->point = 0; }
//
//    int getPoint() { return this->point++; }
//};


#endif //GNN_ACCELERATION_LANGUAGE_COMPUTE_H
