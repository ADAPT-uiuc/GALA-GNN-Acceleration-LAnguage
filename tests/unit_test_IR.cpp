//
// Created by damitha on 4/9/24.
//
#include "common.h"
#include <iostream>

#ifdef TMKL
typedef long long int ind1_t;
#else
typedef uint32_t ind1_t;
#endif

#ifdef TMKL
typedef long long int ind2_t;
#else
typedef uint64_t ind2_t;
#endif
typedef float val_t;
typedef int val_int_t;

// IR classes
#include "../src/codegen/gala/ir/data.h"
#include "../src/codegen/gala/ir/compute.h"

// Code generator
#include "../src/codegen//gala/codegen/cuda.h"

// Matrix classes
//#include "../src/utils/mtx_io.h"
#include "../src/matrix/dense_matrix.h"
#include "../src/matrix/csrc_matrix.h"


//Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DMd_t;
//Dense matrix with integer values.
typedef DenseMatrix<ind1_t, ind2_t, val_int_t> DMi_t;
// Sparse matrix (graph)
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM_t;


int main(int argc, char **argv) {
//    std::cout << "Hello" << std::endl;
//
//    auto g_feats = DataNode<DMd_t>("G.feats", 0, 10);
//    std::cout << "Test IR--1: " << g_feats.getName() << std::endl;
//
//    auto g_graph = DataNode<SM_t>("G.graph", 0, 10);
//    auto graph_data = DataList<SM_t>(true);
//    g_graph.setData(&graph_data);
//    std::cout << "Test IR--2: " << g_graph.getName() << std::endl;
//
//    auto g_relation = RelationEdge<DMd_t, SM_t>(&g_feats, ROW, &g_graph, ROW);
//    std::cout << "Test relation nodes: " << g_relation.getNode1()->getName() << " " << g_relation.getNode2()->getName()
//              << std::endl;
//    std::cout << "Test relation edge: " << g_relation.getRelation1() << " " << g_relation.getRelation2()
//              << std::endl;
//
//    auto g_graph2 = g_graph.cloneData();
//    g_graph2.getData().setIndependence(false);
//    auto g_transform = TransformEdge<SM_t>(&g_graph, &g_graph2);
//
//    auto col_tile_trans = TransformData(COL_TILE);
//    col_tile_trans.addParam("65849");
//    g_transform.addTransformation(&col_tile_trans);
//
//    auto new_compute = StatementNode(AGGREGATE, 0);
//
//    std::cout << "Test transform node: " << g_transform.getNode1()->getName() << std::endl;

    // TODO Create the IR representation for a simple GCN computation.
    //  ideally this should have been lowered to by the front end.
    //   1. Get the basic compute running.
    //   2. Add transformations to the model (Add a data transformation for now -- Tiling)

    // Init point counter
    std::vector<ComputeNode*> program;
    auto pc = PointCounter();
    int now;

    // 1 - Load the graph dataset. Produces the data nodes for the graph and features.
    now = pc.getPoint();
    auto loadCompute = StatementNode(LOAD_OP, now);
    loadCompute.addParam("Reddit");
    // Graph
    auto initialGraph = DataList(CSR_STYPE, true);
    auto graphData = DataNode<SM_t>("gGraph", now, &initialGraph);
    // Feat
    auto initialFeat = DataList(RM_DTYPE, true);
    auto featData = DataNode<DMd_t>("gFeat", now, &initialFeat);
    // Relation -- TODO Ignore relations for now. Use the visitor class design
//    auto graphFeatRel = RelationEdge<SM_t, DMd_t>(&graphData, ROW_RELATION, &featData, ROW_RELATION);
    auto graphFeatRel = RelationEdge(&graphData, ROW_RELATION, &featData, ROW_RELATION);
    loadCompute.addOutputData(&graphData);
    loadCompute.addOutputData(&featData);

    // 2 - Graph transformations
    now = pc.getPoint();
    auto reorderGraph = TransformData(REORDER_TRNS);
    reorderGraph.addParam("rabbit");
    // TODO Ignore for now. Need to implement the sampling operation n how it
    //  works with the mask
    auto sampleGraph = TransformData(SAMPLE_TRNS);
    auto colTileGraph = TransformData(COL_TILE_TRNS);
    colTileGraph.addParam("65000");
    // Update existing graph object
    graphData.setEnd(now);
    // Create new graph object
    auto graphDataTR = graphData.cloneData();
    auto currentLevel = graphDataTR.getData();
    auto newGraphLevel = DataList(CSR_STYPE, false);
    currentLevel.setNext(&newGraphLevel);
    graphDataTR.setStart(now);
    // TODO might need another visitor here
//    auto transformGraph = TransformEdge<SM_t>(&graphData, &graphDataTR);
    auto transformGraph = TransformEdge(&graphData, &graphDataTR);
    transformGraph.addTransformation(&reorderGraph);
    transformGraph.addTransformation(&sampleGraph);
    transformGraph.addTransformation(&colTileGraph);
    auto transformGraphComp = TransformOpNode(now, &transformGraph);

    // Entering the loop
    // 3 - Get degrees from the graph data
    now = pc.getPoint();
    auto degreesCompute1 = StatementNode(DEGREES_OP, now);
    // TODO also store the matrix data here??
    auto initialDegrees1 = DataList(RM_DTYPE, true);
    auto degreesData1 = DataNode<DMd_t>("deg", now, &initialDegrees1);
    degreesCompute1.addInputData(&graphData);
    degreesCompute1.addOutputData(&degreesData1);
    // ****** Loop-start ******
    auto trainingLoop = TrainingLoopNode(100, now);
    trainingLoop.addLoopNode(&degreesCompute1);

    // 4 - Get -0.5 power of degrees to get norm
    now = pc.getPoint();
    auto powCompute1 = StatementNode(POWER_OP, now);
    powCompute1.addParam("-0.5");
    auto initialPow1 = DataList(RM_DTYPE, true);
    auto normData1 = DataNode<DMd_t>("norm", now, &initialPow1);
    powCompute1.addInputData(&degreesData1);
    powCompute1.addOutputData(&normData1);
    trainingLoop.addLoopNode(&powCompute1);

    // 5 - Transform features (Slicing)
    now = pc.getPoint();
    auto sliceFeats = TransformData(COL_TILE_TRNS);
    // TODO Connect to a variable
    sliceFeats.addParam("128");
    // Update existing graph object
    featData.setEnd(now);
    // Create new graph object
    auto featDataTR1 = featData.cloneData();
    auto currentLevelFeat1 = featDataTR1.getData();
    auto newFeatLevel1 = DataList(RM_DTYPE, false);
    currentLevelFeat1.setNext(&newFeatLevel1);
    featDataTR1.setStart(now);
    // TODO might need another visitor here
//    auto transformFeat1 = TransformEdge<DMd_t>(&featData, &featDataTR1);
    auto transformFeat1 = TransformEdge(&featData, &featDataTR1);
    transformFeat1.addTransformation(&sliceFeats);

    // 6 - Apply Edge (SDDMM) -- TODO Ignore for now?
    //  TODO in the code generation component just to a switch type check.
    //   Ignore what happens for apply_edges for now.
    now = pc.getPoint();
    auto applyEdgeCompute1 = StatementNode(APPLY_EDGES_OP, now);
    applyEdgeCompute1.addParam("dsl.fn.mul");
    auto normGraph1 = graphDataTR.cloneData();
    normGraph1.setStart(now);
    applyEdgeCompute1.addInputData(&graphDataTR);
    applyEdgeCompute1.addInputData(&normData1);
    applyEdgeCompute1.addInputData(&normData1);
    applyEdgeCompute1.addOutputData(&normGraph1);
    trainingLoop.addLoopNode(&applyEdgeCompute1);

    // TODO - Ignore compute transformations for now
    // 7 - Aggregate (SpMM)
    now = pc.getPoint();
    auto aggregateCompute1 = StatementNode(AGGREGATE_OP, now);
    aggregateCompute1.addParam("dsl.aggregate.mul_sum");
    auto aggrFeatData1 = featDataTR1.cloneData();
    // TODO setting this should be based on def-use chains
    featDataTR1.setEnd(now);
    aggrFeatData1.setStart(now);
    aggregateCompute1.addInputData(&normGraph1);
    aggregateCompute1.addInputData(&featDataTR1);
    aggregateCompute1.addOutputData(&aggrFeatData1);
    trainingLoop.addLoopNode(&aggregateCompute1);

    // 8 - Update features (GEMM)
    now = pc.getPoint();
    auto updateCompute1 = StatementNode(FFN_OP, now);
    updateCompute1.addParam("256");
    auto ffnFeatData1 = aggrFeatData1.cloneData();
    // TODO setting this should be based on def-use chains
    aggrFeatData1.setEnd(now);
    ffnFeatData1.setStart(now);
    updateCompute1.addInputData(&aggrFeatData1);
    updateCompute1.addOutputData(&ffnFeatData1);
    // Add bias
    now = pc.getPoint();
    auto biasCompute1 = StatementNode(BIAS_OP, now);
    auto biasFeatData1 = ffnFeatData1.cloneData();
    // TODO setting this should be based on def-use chains
    ffnFeatData1.setEnd(now);
    biasFeatData1.setStart(now);
    biasCompute1.addInputData(&ffnFeatData1);
    biasCompute1.addOutputData(&biasFeatData1);
    // Apply non-linear
    now = pc.getPoint();
    auto nlnCompute1 = StatementNode(NON_LNR_OP, now);
    auto nlnFeatData1 = biasFeatData1.cloneData();
    // TODO setting this should be based on def-use chains
    biasFeatData1.setEnd(now);
    nlnFeatData1.setStart(now);
    nlnCompute1.addInputData(&biasFeatData1);
    nlnCompute1.addOutputData(&nlnFeatData1);
    trainingLoop.addLoopNode(&updateCompute1);
    trainingLoop.addLoopNode(&biasCompute1);
    trainingLoop.addLoopNode(&nlnCompute1);
    // TODO Do a single layer for now
    // Exiting the loop

    // Add compute nodes to the program
    program.push_back(&loadCompute);
    program.push_back(&transformGraphComp);
    program.push_back(&trainingLoop);

    // Pass the program along to the IR stages and generate code
    // TODO - Add intermediate transformation stages
    // TODO - Pass to the final code generation

}