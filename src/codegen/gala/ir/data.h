//
// Created by damitha on 4/1/24.
//
#ifndef GNN_ACCELERATION_LANGUAGE_DATA_H
#define GNN_ACCELERATION_LANGUAGE_DATA_H

#include <string>
#include <vector>

/***
 * This IR is only used for code generation. Unless we need to add the characteristics of the data items to be used in
 * something like the cost model, data nodes are just placeholders for the data not the actual data.
 */


// Data format used in the data IR
enum DataFormat {
    // Sparse
    CSR_STYPE, // Compressed sparse row
    CSC_STYPE, // Compressed sparse column
    DCSR_STYPE, // Double compressed sparse row
    COO_STYPE, // Coordinate
    //Dense
    RM_DTYPE, // Row major
    CM_DTYPE, // Column major
    // High-level - Notify the existence of multiple levels in the IR
    Graph, // Graph/Sparse objects
    Dense // Features/Dense objects
};

// Relation between dimensions of the data representations
enum RelationDim {
    ROW_RELATION,
    COL_RELATION,
    ALL_RELATION // Point-wise relation
};

enum TransformationTypes {
    COL_TILE_TRNS, // Matrix - Can be either column tile or slicing
    SAMPLE_TRNS, // Graph - Sample the input graph
    PARTITION_TRNS, // Graph - Partition the graph
    REORDER_TRNS, // Graph - Reorder the graph
    BALANCE_TRNS, // Graph - Load balance the graph
    FORMAT_TRNS, // Matrix - Change the data representation format of the matrix
};

/***
 * Design decisions
 *  - Move the meta-data sharing feature to the code generation. Combine based on the relation nodes.
 *  - Hooks to change things later on by adding data dependant cost models
 *      - Use the name as an ID? (SSA-like)
 *  - TODO How do you know when and where to add the code to transform data?
 *      Based on the input / outputs to the computations.
 *      Traversal of the program happens based on the computation IR,
 *      any additional code added for transformations
 *  - TODO Remove templatization? Currently have multiple input types for dense data, dense labels, and sparse
 *      The current approach will need you to repeat a lot of things?
 */

/***
 * TODO - Find a better name for this. dataNode is a sub element of dataList, but the name makes it sound otherwise
 * @tparam dM - data matrix type info (dense or sparse, with different number types (f32, f64, int, long))
 */
class DataList {
private:
    // If there is a next level, if not null
    DataList *nextLevel;
    // TODO moved the sharing meta-data feature to code generation.
    // List all the data formats that are in the data list (multiple to supports things like ASpT)
    std::vector<DataFormat> formats;
    // If the data items are independent of one another
    bool independent;
public:
    // Constructors
    DataList(DataFormat initialFormat, bool independence) {
        this->independent = independence;
        this->formats.push_back(initialFormat);
        this->nextLevel = nullptr;
    }

    DataList(bool independence) {
        this->independent = independence;
    }

    // Control next level
    bool hasNext() {
        if (this->nextLevel == nullptr) {
            return false;
        }
        return true;
    }

    void setNext(DataList *newLevel) {
        this->nextLevel = newLevel;
    }

    // Formats
    void addFormat(DataFormat format) {
        this->formats.push_back(format);
    }

    // You don't need to remove a single format unless you're making a mistake.
    //  A ClearFormats is necessary when you are creating a new level in the data list.
    void clearFormats() {
        this->formats.clear();
    }

    // Independence
    //  Shows the parallelization independence between data items in the current level.
    bool getIndependence() {
        return this->independent;
    }
    void setIndependence(bool independence) {
        this->independent = independence;
    }
};

class BaseData {
};
//class DataNode: public BaseData

/***
 * Node in the DataIR
 * @tparam dM - data matrix type info (dense or sparse, with different number types (f32, f64, int, long))
 */
template<class dM>
class DataNode: public BaseData{
private:
    // Name of the matrix
    std::string name;

    // Hierarchical data items
    DataList *dataList;

    int startPoint;
    int endPoint;

public:
    // Constructors
    DataNode(std::string name, int start, int end, DataList* newData) {
        this->name = name;
        this->startPoint = start;
        this->endPoint = end;
        this->dataList = newData;
    }
    DataNode(std::string name, int start, DataList* newData) {
        this->name = name;
        this->startPoint = start;
        this->dataList = newData;
    }
    DataNode(std::string name, int start, int end) {
        this->name = name;
        this->startPoint = start;
        this->endPoint = end;
    }

    DataNode(std::string name, int start) {
        this->name = name;
        this->startPoint = start;
    }

    DataNode(std::string name) {
        this->name = name;
    }

    DataNode<dM> cloneData(){
        return DataNode<dM>(this->name, this->startPoint, this->endPoint, this->dataList);
    }

    // getters/setters
    std::string getName() {
        return this->name;
    }

    // Data list
    DataList getData() {
        return this->dataList;
    }

    void setData(DataList* newData) {
        this->dataList = newData;
    }

    // Get the start and end points
    int getStart() {
        return this->startPoint;
    }

    int getEnd() {
        return this->endPoint;
    }

    // Change the start and end points
    void setStart(int newStart) {
        this->startPoint = newStart;
    }

    void setEnd(int newEnd) {
        this->endPoint = newEnd;
    }
};

class DataEdge {
private:
    BaseData *node1;
    BaseData *node2;
public:
    DataEdge(BaseData *n1, BaseData *n2) {
        this->node1 = n1;
        this->node2 = n2;
    }

    // No need for setters? The relation should not change at any time as time goes not
    BaseData *getNode1() {
        return node1;
    }

    BaseData *getNode2() {
        return node2;
    }
};

class RelationEdge : public DataEdge {
private:
    RelationDim rel1;
    RelationDim rel2;
public:
    RelationEdge(BaseData *n1, RelationDim r1, BaseData *n2, RelationDim r2) : DataEdge(n1, n2) {
        this->rel1 = r1;
        this->rel2 = r2;
    }

    // Only have the getters for the relations
    RelationDim getRelation1() {
        return rel1;
    }
    RelationDim getRelation2() {
        return rel2;
    }
};


class TransformData{
private:
    TransformationTypes transformation;
    std::vector<std::string> params;
public:
    TransformData(TransformationTypes trns){
        this->transformation = trns;
    }

    TransformationTypes getTransformation(){
        return this->transformation;
    }
    std::vector<std::string>* getParams(){
        return &this->params;
    }

    void addParam(std::string param){
        this->params.push_back(param);
    }
};

class TransformEdge : public DataEdge{
private:
    std::vector<TransformData*> transformations;
public:
    TransformEdge(BaseData *n1, BaseData *n2) : DataEdge(n1, n2) {
    }

    // Only have the getters for the relations
    void addTransformation(TransformData* trns) {
        transformations.push_back(trns);
    }
};

// ------------------- Pre-base class -------------------------
//
//template<class dM1, class dM2>
//class DataEdge {
//private:
//    DataNode<dM1> *node1;
//    DataNode<dM2> *node2;
//public:
//    DataEdge(DataNode<dM1> *n1, DataNode<dM2> *n2) {
//        this->node1 = n1;
//        this->node2 = n2;
//    }
//
//    // No need for setters? The relation should not change at any time as time goes not
//    DataNode<dM1> *getNode1() {
//        return node1;
//    }
//
//    DataNode<dM2> *getNode2() {
//        return node2;
//    }
//};
//
//template<class dM1, class dM2>
//class RelationEdge : public DataEdge<dM1, dM2> {
//private:
//    RelationDim rel1;
//    RelationDim rel2;
//public:
//    RelationEdge(DataNode<dM1> *n1, RelationDim r1, DataNode<dM2> *n2, RelationDim r2) : DataEdge<dM1, dM2>(n1, n2) {
//        this->rel1 = r1;
//        this->rel2 = r2;
//    }
//
//    // Only have the getters for the relations
//    RelationDim getRelation1() {
//        return rel1;
//    }
//    RelationDim getRelation2() {
//        return rel2;
//    }
//};
//
//
//class TransformData{
//private:
//    TransformationTypes transformation;
//    std::vector<std::string> params;
//public:
//    TransformData(TransformationTypes trns){
//        this->transformation = trns;
//    }
//
//    TransformationTypes getTransformation(){
//        return this->transformation;
//    }
//    std::vector<std::string>* getParams(){
//        return &this->params;
//    }
//
//    void addParam(std::string param){
//        this->params.push_back(param);
//    }
//};
//
//template<class dM1>
//class TransformEdge : public DataEdge<dM1, dM1> {
//private:
//    std::vector<TransformData*> transformations;
//public:
//    TransformEdge(DataNode<dM1> *n1, DataNode<dM1> *n2) : DataEdge<dM1, dM1>(n1, n2) {
//    }
//
//    // Only have the getters for the relations
//    void addTransformation(TransformData* trns) {
//        transformations.push_back(trns);
//    }
//};


#endif //GNN_ACCELERATION_LANGUAGE_DATA_H
