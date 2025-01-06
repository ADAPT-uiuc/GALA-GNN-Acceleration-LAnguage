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
#include "../src/ir/data.h"
#include "../src/ir/compute.h"
#include "../src/codegen/cuda.h"

// Code generator
//#include "../src/codegen//gala/codegen/cuda.h"

// Matrix classes
//#include "../src/utils/mtx_io.h"
#include "../src/formats/dense_matrix.h"
#include "../src/formats/csrc_matrix.h"


//Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DMd_t;
//Dense matrix with integer values.
typedef DenseMatrix<ind1_t, ind2_t, val_int_t> DMi_t;
// Sparse matrix (graph)
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM_t;


int main(int argc, char **argv) {
    // TODO Create the IR representation for a simple GCN computation.
    //  ideally this should have been lowered to by the front end.
    //   1. Get the basic compute running.
    //   2. Add transformations to the model (Add a data transformation for now -- Tiling)

	/** RULES -- res input is always the 1st input for a computaiton op */

    // Init point counter
    std::vector<CIRNode*> program;
	std::vector<RelationEdge*> dependencies;
	std::vector<RelationEdge*> associations;
	std::vector<TransformEdge*> transforms;

    auto loadDataset = ForwardNode(POINTWISE, LOAD_OP);
    loadDataset.addParam("/shared/damitha2/gala_npy/RedditDataset/");
    // Graph
    auto graphInfo = DataInfo(CSR_STYPE, false, false);
    auto rootGraphLevel = DataLevel(&graphInfo, true);
    auto graphData = DataNode("Graph", INT32, INT32, F32, &rootGraphLevel);
    // Feat
    auto featInfo = DataInfo(RM_DTYPE);
    featInfo.setDims(-1, -2);
    auto rootFeatLevel = DataLevel(&featInfo, true);
    auto featData = DataNode("Feat", INT32, INT32, F32, &rootFeatLevel);

	// Association between graph and features
	auto graphFeatAssociation = RelationEdge(&graphData, ALL_RELATION, &featData, ROWS_RELATIOM);
	associations.push_back(&graphFeatAssociation);
	loadDataset.addOutputData(&featData);
	loadDataset.addOutputData(&graphData);

	auto originalRootGraphLevel = graphData.getData(); // Returns pointer
	auto originalGraphInfo = originalRootGraphLevel->next(); // Returns pointer
	auto transformedGraphInfo = DataInfo(CSR_STYPE, true, true);
	transformedGraphInfo.addOpt(COL_TILE_DOPT, "65000");
	auto transformedTileGraphLevel = DataLevel(&transformedGraphInfo, false);
	auto transformedRootGraphLevel = DataLevel(&transformedTileGraphLevel, true);
	auto transformedGraph = DataNode("Graph-Tile", graphData.getIType(), graphData.getNType(),
		graphData.getVType(), &transformedRootGraphLevel);
	// Association between graph and features
	auto trgrapgFeatAssociation = RelationEdge(&transformedGraph, ALL_RELATION, &featData, ROWS_RELATIOM);
	associations.push_back(&graphFeatAssociation);
	auto tileTransformation = TransformData(COL_TILE_DOPT);
	tileTransformation.addParam("65000");
	auto graphTrgraph = TransformEdge(&graphData, &transformedGraph);
	graphTrgraph.addTransformation(&tileTransformation);
	transforms.push_back(&graphTrgraph);

	featInfo.setDims(-1, 605);

	auto trainingLoop = TrainingLoopNode(100);

    // Add aggregate operation
    auto aggregate = ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_OP);
    auto outputInfo = DataInfo(RM_DTYPE);
    outputInfo.setDims(-1, 605); // -1=N=232965, the number of nodes in the graph
    auto rootOutputLevel = DataLevel(&outputInfo, true);
    auto outputData = DataNode("Out1", INT32, INT32, F32, &rootOutputLevel);
	aggregate.addInputData(&featData);
	aggregate.addInputData(&transformedGraph);
    aggregate.addOutputData(&outputData);
    trainingLoop.addLoopNode(&aggregate);

    // Dependency relation between the features and the aggregated output
	auto inOutAggrRelationFeat = RelationEdge(&featData, ALL_RELATION, &outputData, ALL_RELATION);
	// Dependency relation between the graph and the aggregated output
	auto inOutAggrRelationGraph = RelationEdge(&transformedGraph, ALL_RELATION, &outputData, ROWS_RELATION);
    dependencies.push_back(&inOutAggrRelationFeat);
    dependencies.push_back(&inOutAggrRelationGraph);

    // Add weight operation
    auto ffn = ForwardNode(UPDATE_NODE, FFN_OP);
    // Weight as a matrix in the DIR
    auto weightInfo = DataInfo(CM_DTYPE);
    weightInfo.setDims(-2, -3); // -2=input embedding dimension, -3=output classes
    auto weightLevel = DataLevel(&weightInfo, true);
    auto weightData = DataNode("Weight1", INT32, INT32, F32, &weightLevel);
    // Res DIR
    auto resInfo = DataInfo(RM_DTYPE);
    resInfo.setDims(-1, -3); // -1=N=232965, the number of nodes in the graph, -3=output classes

    // set dimenions from the new schedule information
    weightInfo.setDims(605, 41); //
    resInfo.setDims(-1, 41); // -1=N=232965, the number of nodes in the graph

    auto rootResLevel = DataLevel(&outputInfo, true);
    auto resData = DataNode("Res1", INT32, INT32, F32, &rootResLevel);
    aggregate.addInputData(&outputData);
    aggregate.addInputData(&weightData);
    aggregate.addOutputData(&resData);
    aggregate.addOpt(COARSE_COPT, 4);
    trainingLoop.addLoopNode(&ffn);
    auto inOutWeightDepRelationFeat = RelationEdge(&outputData, ALL_RELATION, &resData, ALL_RELATION);
    auto inOutWeightDepRelationWeight = RelationEdge(&weightData, COLS_RELATION, &resData, ROWS_RELATION);
    dependencies.push_back(&inOutWeightDepRelationFeat);
    dependencies.push_back(&inOutWeightDepRelationWeight);
    auto inOutWeightAssociation = RelationEdge(&outputData, ROWS_RELATION, &weightData, COLS_RELATION);
    associations.push_back(&inOutWeightAssociation);

    // The entire program
    program.push_back(&loadDataset);
	program.push_back(&trainingLoop);

	auto ctx = new GALAContext(GPU_DEVICE, SINGLE_NODE_SINGLE);
	std::string outputPath = "../test-codegen/";
	auto genCode = CUDAGenerator(ctx, outputPath);
	genCode.writeCode(program);

    // Should be enough for now
	std::cout << "Works!" << std::endl;


//    // Relation -- TODO Ignore relations for now. Use the visitor class design
////    auto graphFeatRel = RelationEdge<SM_t, DMd_t>(&graphData, ROW_RELATION, &featData, ROW_RELATION);
//    auto graphFeatRel = RelationEdge(&graphData, ROW_RELATION, &featData, ROW_RELATION);
//    loadCompute.addOutputData(&graphData);
//    loadCompute.addOutputData(&featData);
//
//    // 2 - Graph transformations
//    now = pc.getPoint();
//    auto reorderGraph = TransformData(REORDER_TRNS);
//    reorderGraph.addParam("rabbit");
//    // TODO Ignore for now. Need to implement the sampling operation n how it
//    //  works with the mask
//    auto sampleGraph = TransformData(SAMPLE_TRNS);
//    auto colTileGraph = TransformData(COL_TILE_TRNS);
//    colTileGraph.addParam("65000");
//    // Update existing graph object
//    graphData.setEnd(now);
//    // Create new graph object
//    auto graphDataTR = graphData.cloneData();
//    auto currentLevel = graphDataTR.getData();
//    auto newGraphLevel = DataList(CSR_STYPE, false);
//    currentLevel.setNext(&newGraphLevel);
//    graphDataTR.setStart(now);
//    // TODO might need another visitor here
////    auto transformGraph = TransformEdge<SM_t>(&graphData, &graphDataTR);
//    auto transformGraph = TransformEdge(&graphData, &graphDataTR);
//    transformGraph.addTransformation(&reorderGraph);
//    transformGraph.addTransformation(&sampleGraph);
//    transformGraph.addTransformation(&colTileGraph);
//    auto transformGraphComp = TransformOpNode(now, &transformGraph);
//
//    // Entering the loop
//    // 3 - Get degrees from the graph data
//    now = pc.getPoint();
//    auto degreesCompute1 = StatementNode(DEGREES_OP, now);
//    // TODO also store the matrix data here??
//    auto initialDegrees1 = DataList(RM_DTYPE, true);
//    auto degreesData1 = DataNode("deg", UINT32, UINT64, INT32, now, &initialDegrees1);
//    degreesCompute1.addInputData(&graphData);
//    degreesCompute1.addOutputData(&degreesData1);
//    // ****** Loop-start ******
//    auto trainingLoop = TrainingLoopNode(100, now);
//    trainingLoop.addLoopNode(&degreesCompute1);
//
//    // 4 - Get -0.5 power of degrees to get norm
//    now = pc.getPoint();
//    auto powCompute1 = StatementNode(POWER_OP, now);
//    powCompute1.addParam("-0.5");
//    auto initialPow1 = DataList(RM_DTYPE, true);
//    auto normData1 = DataNode("norm", UINT32, UINT64, F32, now, &initialPow1);
//    powCompute1.addInputData(&degreesData1);
//    powCompute1.addOutputData(&normData1);
//    trainingLoop.addLoopNode(&powCompute1);
//
//    // 5 - Transform features (Slicing)
//    now = pc.getPoint();
//    auto sliceFeats = TransformData(COL_TILE_TRNS);
//    // TODO Connect to a variable
//    sliceFeats.addParam("128");
//    // Update existing graph object
//    featData.setEnd(now);
//    // Create new graph object
//    auto featDataTR1 = featData.cloneData();
//    auto currentLevelFeat1 = featDataTR1.getData();
//    auto newFeatLevel1 = DataList(RM_DTYPE, false);
//    currentLevelFeat1.setNext(&newFeatLevel1);
//    featDataTR1.setStart(now);
//    // TODO might need another visitor here
////    auto transformFeat1 = TransformEdge<DMd_t>(&featData, &featDataTR1);
//    auto transformFeat1 = TransformEdge(&featData, &featDataTR1);
//    transformFeat1.addTransformation(&sliceFeats);
//
//    // 6 - Apply Edge (SDDMM) -- TODO Ignore for now?
//    //  TODO in the code generation component just to a switch type check.
//    //   Ignore what happens for apply_edges for now.
//    now = pc.getPoint();
//    auto applyEdgeCompute1 = StatementNode(APPLY_EDGES_OP, now);
//    applyEdgeCompute1.addParam("dsl.fn.mul");
//    auto normGraph1 = graphDataTR.cloneData();
//    normGraph1.setStart(now);
//    applyEdgeCompute1.addInputData(&graphDataTR);
//    applyEdgeCompute1.addInputData(&normData1);
//    applyEdgeCompute1.addInputData(&normData1);
//    applyEdgeCompute1.addOutputData(&normGraph1);
//    trainingLoop.addLoopNode(&applyEdgeCompute1);
//
//    // TODO - Ignore compute transformations for now
//    // 7 - Aggregate (SpMM)
//    now = pc.getPoint();
//    auto aggregateCompute1 = StatementNode(AGGREGATE_OP, now);
//    aggregateCompute1.addParam("dsl.aggregate.mul_sum");
//    auto aggrFeatData1 = featDataTR1.cloneData();
//    // TODO setting this should be based on def-use chains
//    featDataTR1.setEnd(now);
//    aggrFeatData1.setStart(now);
//    aggregateCompute1.addInputData(&normGraph1);
//    aggregateCompute1.addInputData(&featDataTR1);
//    aggregateCompute1.addOutputData(&aggrFeatData1);
//    trainingLoop.addLoopNode(&aggregateCompute1);
//
//    // 8 - Update features (GEMM)
//    now = pc.getPoint();
//    auto updateCompute1 = StatementNode(FFN_OP, now);
//    updateCompute1.addParam("256");
//    auto ffnFeatData1 = aggrFeatData1.cloneData();
//    // TODO setting this should be based on def-use chains
//    aggrFeatData1.setEnd(now);
//    ffnFeatData1.setStart(now);
//    updateCompute1.addInputData(&aggrFeatData1);
//    updateCompute1.addOutputData(&ffnFeatData1);
//    // Add bias
//    now = pc.getPoint();
//    auto biasCompute1 = StatementNode(BIAS_OP, now);
//    auto biasFeatData1 = ffnFeatData1.cloneData();
//    // TODO setting this should be based on def-use chains
//    ffnFeatData1.setEnd(now);
//    biasFeatData1.setStart(now);
//    biasCompute1.addInputData(&ffnFeatData1);
//    biasCompute1.addOutputData(&biasFeatData1);
//    // Apply non-linear
//    now = pc.getPoint();
//    auto nlnCompute1 = StatementNode(NON_LNR_OP, now);
//    auto nlnFeatData1 = biasFeatData1.cloneData();
//    // TODO setting this should be based on def-use chains
//    biasFeatData1.setEnd(now);
//    nlnFeatData1.setStart(now);
//    nlnCompute1.addInputData(&biasFeatData1);
//    nlnCompute1.addOutputData(&nlnFeatData1);
//    trainingLoop.addLoopNode(&updateCompute1);
//    trainingLoop.addLoopNode(&biasCompute1);
//    trainingLoop.addLoopNode(&nlnCompute1);
//    // TODO Do a single layer for now
//    // Exiting the loop
//
//    // Add compute nodes to the program
//    program.push_back(&loadCompute);
//    program.push_back(&transformGraphComp);
//    program.push_back(&trainingLoop);
//
//    // Pass the program along to the IR stages and generate code
//    // TODO - Add intermediate transformation stages
//    // TODO - Pass to the final code generation
//    auto context = GALAContext(GPU_DEVICE, SINGLE_NODE_SINGLE);
//    std::string outPath = "/home/damitha/GNN-Acceleration-Language/test.py";
//
//    auto gen = GenerateCUDA(&context, &outPath);
//    gen.generateCode(program);
//    gen.closeStream();
}