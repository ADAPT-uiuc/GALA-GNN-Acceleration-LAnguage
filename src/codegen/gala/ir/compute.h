//
// Created by damitha on 4/1/24.
//

#ifndef GNN_ACCELERATION_LANGUAGE_COMPUTE_H
#define GNN_ACCELERATION_LANGUAGE_COMPUTE_H

#include <any>
//#include <memory>
#include "data.h"

enum ComputeOp {
    LOAD_OP,
    DEGREES_OP,
    POWER_OP,
    APPLY_EDGES_OP, // SDDMM
    AGGREGATE_OP, // SpMM
    FFN_OP, // Can also be UPDATE
    BIAS_OP,
    NON_LNR_OP,
    // Control statements
    IF_CONTROL,
    TRAIN_CONTROL,
    // Transformation
    TRANSFORM_OP
};

// TODO - Add node / edge aggregation types.

class ComputeNode {
private:
    ComputeOp op;

    // Program start point
    int point;
public:
    ComputeNode(ComputeOp op, int point) {
        this->op = op;
        this->point = point;
    }

    // Op shouldn't change

    // Point
    int getPoint() { return this->point; }

    void setPoint(int new_point) { this->point = new_point; }

    // Op -- Also used to identify between the
    ComputeOp getOp() { return this->op; }
};

class StatementNode : public ComputeNode {
private:
    // Parameters to be passed along to the function
    std::vector <string> params;

    // Can't have an array with templatized arguments.
    //  Just store the pointers for the multiple data objects?
    //  Then just convert to the correct type when you're using it.
    //  Still need to know the type of the matrices that you're lowering to
//    std::vector<std::any*> inputData;
//    std::vector<std::any*> outputData;

    std::vector<DataNode *> inputData;
    std::vector<DataNode *> outputData;
public:
    StatementNode(ComputeOp op, int point) : ComputeNode(op, point) {}

    // Op shouldn't change

    // Params can only add new (No need to remove any)
    void addParam(std::string new_param) { this->params.push_back(new_param); }

    // At Nihility's(IX) end
    std::string getParam(int ix) { return this->params.at(ix); }

    // Data items should be constant once added (No need to remove elements)
    void addInputData(DataNode *new_input) { this->inputData.push_back(new_input); }

    void addOutputData(DataNode *new_output) { this->outputData.push_back(new_output); }
};

// TODO Need to have methods to manage control flow -- Not sure if this node will be used or not. (Archive for now?)
//  Have separate classes for the control structures
class IfNode : public ComputeNode {
private:
    std::string condition;
    std::vector<ComputeNode *> truePath;
    std::vector<ComputeNode *> elsePath;

public:
    IfNode(int point, std::string condition) : ComputeNode(IF_CONTROL, point) { this->condition = condition; }

    std::string getCondition() { return condition; }

    void addTrue(ComputeNode *newNode) { this->truePath.push_back(newNode); }

    void addElse(ComputeNode *newNode) { this->elsePath.push_back(newNode); }
};

class TrainingLoopNode : public ComputeNode {
private:
    int numIter;
    int stepValid; // The steps in the program validation happens. If not specified the numIter.
    // TODO need an easy way to find a node by an ID and then return + remove it.
    //  temp solution - Remove everything and then add everything back
    std::vector<ComputeNode *> loop;
public:
    TrainingLoopNode(int point, int numIter) : ComputeNode(TRAIN_CONTROL, point) {
        this->numIter = numIter;
        this->stepValid = numIter;
    }

    TrainingLoopNode(int point, int numIter, int stepValid) : ComputeNode(TRAIN_CONTROL, point) {
        this->numIter = numIter;
        this->stepValid = stepValid;
    }

    int getIter() { return this->numIter; }

    int getValidStep() { return this->stepValid; }

    void clearLoopNodes() { this->loop.clear(); }

    void addLoopNode(ComputeNode *newNode) { this->loop.push_back(newNode); }

    int getLoopNodeNum() { return this->loop.size(); }

    std::vector<ComputeNode *> *getLoopNodes() { return &loop; }

    ComputeNode *getNode(int i) { return this->loop.at(i); }
};

class TransformOpNode : public ComputeNode {
private:
    TransformEdge *transformData;
public:
    TransformOpNode(int point, TransformEdge *trEdge) : ComputeNode(TRANSFORM_OP, point) {
        this->transformData = trEdge;
    }

    TransformEdge *getTransformations() { return this->transformData; }
};

class PointCounter {
private:
    int point;
public:
    PointCounter() { this->point = 0; }

    int getPoint() { return this->point++; }
};


#endif //GNN_ACCELERATION_LANGUAGE_COMPUTE_H
