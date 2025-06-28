//

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
    // Tensor
    RM_DTYPE, // Row major
    CM_DTYPE, // Column major
    // High-level - Notify the existence of multiple levels in the IR
    Graph, // Graph/Sparse objects
    Tensor, // Features/Dense objects
  	// Scalar / Constant values
  	STR, // String
  	NUM, // Number
};

// Relation between dimensions of the data representations
enum RelationDim {
    ROWS_RELATION,
    COLS_RELATION,
    ALL_RELATION // Point-wise relation
};

// enum RelationType {
//   DEPENDENCY_RELATION, // Data dependency
//   TRANSFORMATION_RELATION, // Data transformations
//   ASSOCIATION_RELATION // Associaitons between data (ex -  rows of dense input matrix to rows/cols of a graph)
// };

enum DataOptimization {
    COL_TILE_DOPT, // Column tile a graph (TODO or slice a tensor?)
    SAMPLE_DOPT, // Sample from a given graph
    SUBGRAPH_DOPT, // Subgraph creation
    // TODO the optimization types below are planned but not implmeneted yet
//    PARTITION_DOPT, // Graph - Partition the graph
//    REORDER_DOPT, // Graph - Reorder the graph
//    BALANCE_DOPT, // Graph - Load balance the graph
};

enum NumTypes {
    // Ints
    INT8,
    INT16,
    INT32,
    INT64,
    // UInts
    UINT32,
    UINT64,
    // Floats
    F32,
    F64
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
 *  - Remove templatization? Currently have multiple input types for dense data, dense labels, and sparse
 *      The current approach will need you to repeat a lot of things?
 *     - Have a type attribute for now. Because templetazing would have issues with overrriding etc.
 */

// A base abstract class that can either be a data info (gives information of a data node, is the final/leaf level in a data node
class DataItem {
 public:
   // Function to get if the current data item is a data level or data info.
   virtual bool isLevel() {return false;};
};


class DataInfo: virtual public DataItem {
private:
  DataFormat format;
  std::vector<std::pair<DataOptimization, std::string>> opts;
  bool isDirected;
  bool isWeighted;
  bool isSparse = false;
  int dimRow;
  int dimCol;
    // To be used by the global meta-data array
    int globalIndex;
    // To be used by the global meta-data array
    int defaultIndex = 0;
    std::string defaultName = "";
    bool defaultDirected = false;
    bool isDerived = false;
public:
  // Set the data informaiton
  DataInfo(DataFormat format,
           bool isDirected,
           bool isWeighted){
    this->format = format;
    this->isDirected = isDirected;
    this->isWeighted = isWeighted;
      this->globalIndex = -1;
  };
  DataInfo(DataFormat format){
    this->format = format;
      this->globalIndex = -1;
  };

  // Get opts
  std::vector<std::pair<DataOptimization, std::string>>* getOpts() {
    return &opts;
  };
  // Add opts
  void addOpt(DataOptimization opt, std::string param) {
    opts.push_back(std::make_pair(opt, param));
  };

  // Get and set directed
  bool getDirected() {
    return this->isDirected;
  }
  void setDirected(bool directed) {
    this->isDirected = directed;
  }

  // Get and set if sparse
  bool getSparse()
  {
      return this->isSparse;
  }
  void setSparse(bool sparse)
  {
      this->isSparse = sparse;
  }

  // Get and set weighted
  bool getWeighted() {
    return this->isWeighted;
  }
  void setWeighted(bool weighted) {
    this->isWeighted = weighted;
  }

  void setDims(int dimRow, int dimCol) {
    this->dimRow = dimRow;
    this->dimCol = dimCol;
  }
  int getDimRow() {
    return this->dimRow;
  }
  int getDimCol() {
    return this->dimCol;
  }

  // Information is not a level
  bool isLevel(){
    return false;
  }

  DataFormat getFormat(){
      return this->format;
  }

    void setIndex(int index)
  {
      this->globalIndex = index;
  }
    int getIndex()
  {
      return this->globalIndex;
  }
    void setDefaultIndex(int index)
  {
      this->defaultIndex = index;
  }
    int getDefaultIndex()
  {
      return this->defaultIndex;
  }
    void setDefaultName(std::string name)
  {
      this->defaultName = name;
  }
    std::string getDefaultName()
  {
      return this->defaultName;
  }
    void setDefaultDirected(bool isDirected)
  {
      this->defaultDirected = isDirected;
  }
    bool getDefaultDirected()
  {
      return this->defaultDirected;
  }
    void setDerived(bool deriv){
      this->isDerived = deriv;
  }
    bool getDerived(){
      return this->isDerived;
  }



};

class DataLevel: virtual public DataItem {
private:
    // If there is a next level, if not null
    DataItem *nextItem;
    // If the data items are independent of one another
    bool independent;
public:
    // Constructors
    DataLevel(DataItem* item = nullptr, bool independence = false) {
      this->nextItem = item;
      this->independent = independence;
    }

    // Data level is a data level
    bool isLevel(){
        return true;
    }

    // Control next level
    bool hasNext() {
        if (this->nextItem == nullptr) {
            return false;
        }
        return true;
    }
    DataItem* next(){
        if (this->hasNext()) {
          return this->nextItem;
        } else {
          return nullptr;
        }
    }
    void setNext(DataItem *newLevel) { this->nextItem = newLevel; }


    // Independence
    //  Shows the parallelization independence between data items in the current level.
    bool getIndependence() { return this->independent; }
    void setIndependence(bool independence) { this->independent = independence; }
};



/***
 * Node in the DataIR
 * @tparam dM - data matrix type info (dense or sparse, with different number types (f32, f64, int, long))
 */
class DataNode {
private:
    // Name of the matrix
    std::string name;

    NumTypes indexType;
    NumTypes edgeType;
    NumTypes valueType;

    // Hierarchical data items
    DataLevel *rootLevel;

    // TODO Having all the dependencies as a separate edge list might be better
    // // Relations
    // std::vector<std::pair<DataNode*, RelationDim>> dependencyNodes; // from the
    // std::vector<std::pair<DataNode*, RelationDim>> transformationNodes; // from the transformed data
    // std::vector<std::pair<DataNode*, RelationDim>> associationNodes; // bi-directional

public:
    // Constructors
    DataNode(std::string name, NumTypes iT, NumTypes nT, NumTypes vT, DataLevel *newData) {
        this->name = name;
        this->rootLevel = newData;

        this->indexType = iT;
        this->edgeType = nT;
        this->valueType = vT;
    }

    DataNode cloneNew(std::string newName) {
        DataLevel newRootLevel = DataLevel();
        DataLevel originalRootLevel= newRootLevel;
        DataLevel newBaseLevel;
        newRootLevel.setIndependence(this->rootLevel->getIndependence());

        DataItem* nextLevel = this->rootLevel->next();
        auto currentLevel = dynamic_cast<DataLevel*>(this->rootLevel);
        while (currentLevel)
        {
            newBaseLevel = DataLevel();
            newBaseLevel.setIndependence(currentLevel->getIndependence());
            newRootLevel.setNext(&newBaseLevel);
            newRootLevel = newBaseLevel;

            nextLevel = currentLevel->next();
            currentLevel = dynamic_cast<DataLevel*>(nextLevel);
        }
        auto currentInfo = dynamic_cast<DataInfo*>(nextLevel);
        auto newDataInfo = DataInfo(currentInfo->getFormat(), currentInfo->getWeighted(), currentInfo->getDirected());
        newRootLevel.setNext(&newDataInfo);
        return DataNode(newName,
            this->indexType,
            this->edgeType,
            this->valueType,
            &originalRootLevel);
    }
    DataNode cloneData() {
        return DataNode(this->name, this->indexType, this->edgeType, this->valueType,
                        this->rootLevel);
    }

    // getters/setters
    std::string getName() { return this->name; }
    void setName(std::string newName) { this->name = newName; }

    // Data level
    DataLevel* getData() { return this->rootLevel; }
    void setData(DataLevel *newData) { this->rootLevel = newData; }

    NumTypes getIType() { return this->indexType; }

    NumTypes getNType() { return this->edgeType; }

    NumTypes getVType() { return this->valueType; }

    DataInfo* getDataInfo()
    {
        // Loop till you get the data info
        DataItem* nextLevel = this->rootLevel;
        auto currentLevel = dynamic_cast<DataLevel*>(this->rootLevel);
        while (currentLevel)
        {
            nextLevel = currentLevel->next();
            currentLevel = dynamic_cast<DataLevel*>(nextLevel);
        }
        return dynamic_cast<DataInfo*>(nextLevel);
    }
};

// How to access an edge?
class DataEdge {
private:
    DataNode *node1;
    DataNode *node2;
public:
    DataEdge(DataNode *n1, DataNode *n2) {
        this->node1 = n1;
        this->node2 = n2;
    }

    // No need for setters? The relation should not change at any time
    DataNode *getNode1() { return node1; }
    DataNode *getNode2() { return node2; }
};

class RelationEdge : public DataEdge {
private:
    RelationDim rel1;
    RelationDim rel2;
public:
    RelationEdge(DataNode *n1, RelationDim r1, DataNode *n2, RelationDim r2) : DataEdge(n1, n2) {
        this->rel1 = r1;
        this->rel2 = r2;
    }

    // Only have the getters for the relations
    RelationDim getRelation1() { return rel1; }
    RelationDim getRelation2() { return rel2; }
};


class TransformData {
private:
    DataOptimization transformation;
    std::vector<std::string> params;
public:
    TransformData(DataOptimization trns) { this->transformation = trns; }

    DataOptimization getTransformation() { return this->transformation; }

    std::vector<std::string> *getParams() { return &this->params; }
    std::string getParam(int ix) { return this->params.at(ix); }
    int getNumParam() { return (int)this->params.size(); }
    void addParam(std::string param) { this->params.push_back(param); }
};

class TransformEdge : public DataEdge {
private:
    std::vector<TransformData *> transformations;
public:
    TransformEdge(DataNode *n1, DataNode *n2) : DataEdge(n1, n2) {}

    // Only have the getters for the relations
    void addTransformation(TransformData *trns) { transformations.push_back(trns); }
    TransformData* getTransformation(int ix) { return this->transformations.at(ix); }
    int getNumTransformations() { return this->transformations.size(); }
};

#endif //GNN_ACCELERATION_LANGUAGE_DATA_H
