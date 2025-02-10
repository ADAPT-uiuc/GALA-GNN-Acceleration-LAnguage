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
#include "../src/middle-end/middle-end.h"

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


// TODO Create the IR representation for a simple GCN computation.
//  ideally this should have been lowered to by the front end.
//   1. Get the basic compute running.
//   2. Add transformations to the model (Add a data transformation for now -- Tiling)

/** RULES -- res input is always the 1st input for a computaiton op */

int main(int argc, char **argv) {
    // Init point counter
    std::vector<CIRNode*> program;
	std::vector<RelationEdge*> dependencies;
	std::vector<RelationEdge*> associations;
	std::vector<TransformEdge*> transforms;

    auto loadDataset = ForwardNode(POINTWISE, LOAD_OP);
    loadDataset.addParam("/shared/damitha2/gala_npy/RedditDataset/");

    // Graph
    auto graphInfo = DataInfo(CSR_STYPE, true, true);
    auto rootGraphLevel = DataLevel(&graphInfo, true);
    auto graphData = DataNode("adj0", INT32, INT32, F32, &rootGraphLevel);
    // Feat
    auto featInfo = DataInfo(RM_DTYPE);
    featInfo.setDims(-1, -2);
    auto rootFeatLevel = DataLevel(&featInfo, true);
    auto featData = DataNode("feat", INT32, INT32, F32, &rootFeatLevel);

	// Association between graph and features
	auto graphFeatAssociation = RelationEdge(&graphData, ALL_RELATION, &featData, ROWS_RELATION);
	associations.push_back(&graphFeatAssociation);
	loadDataset.addOutputData(&featData);
	loadDataset.addOutputData(&graphData);

	auto originalRootGraphLevel = graphData.getData(); // Returns pointer
	auto originalGraphInfo = originalRootGraphLevel->next(); // Returns pointer
	auto transformedGraphInfo = DataInfo(CSR_STYPE, false, false);
	transformedGraphInfo.addOpt(COL_TILE_DOPT, "65000");
	auto transformedTileGraphLevel = DataLevel(&transformedGraphInfo, false);
	auto transformedRootGraphLevel = DataLevel(&transformedTileGraphLevel, true);
	auto transformedGraph = DataNode("graph_tile", graphData.getIType(), graphData.getNType(), graphData.getVType(), &transformedRootGraphLevel);
	// Association between graph and features
	auto trgrapgFeatAssociation = RelationEdge(&transformedGraph, ALL_RELATION, &featData, ROWS_RELATION);
	associations.push_back(&graphFeatAssociation);
	auto tileTransformation = TransformData(COL_TILE_DOPT);
	tileTransformation.addParam("65000");
	auto graphTrgraph = TransformEdge(&graphData, &transformedGraph);
	graphTrgraph.addTransformation(&tileTransformation);
	transforms.push_back(&graphTrgraph);

	featInfo.setDims(-1, 605);

	auto trainingLoop = TrainingLoopNode(100);

	// Add weight operation
	auto ffn = ForwardNode(UPDATE_NODE, FFN_OP);
	// Weight as a matrix in the DIR
	auto weightInfo = DataInfo(CM_DTYPE);
	weightInfo.setDims(-2, 32); // -2=input embedding dimension, -3=output classes
	auto weightLevel = DataLevel(&weightInfo, true);
	auto weightData = DataNode("weight1", INT32, INT32, F32, &weightLevel);
	// Res DIR
	auto resInfo = DataInfo(RM_DTYPE);
	resInfo.setDims(-1, 32); // -1=N=232965, the number of nodes in the graph, -3=output classes
	auto rootResLevel = DataLevel(&resInfo, true);
	auto resData = DataNode("res", INT32, INT32, F32, &rootResLevel);
	// set dimenions from the new schedule information
	weightInfo.setDims(605, 32); //
	resInfo.setDims(-1, 32); // -1=N=232965, the number of nodes in the graph
	ffn.addInputData(&featData);
	ffn.addInputData(&weightData);
	ffn.addOutputData(&resData);
	trainingLoop.addLoopNode(&ffn);
	//* Dependencies
	auto inOutWeightDepRelationFeat = RelationEdge(&featData, ALL_RELATION, &resData, ALL_RELATION);
	auto inOutWeightDepRelationWeight = RelationEdge(&weightData, COLS_RELATION, &resData, ROWS_RELATION);
	dependencies.push_back(&inOutWeightDepRelationFeat);
	dependencies.push_back(&inOutWeightDepRelationWeight);
	auto inOutWeightAssociation = RelationEdge(&featData, ROWS_RELATION, &weightData, COLS_RELATION);
	associations.push_back(&inOutWeightAssociation);

	// Add attention weight operation (L side)
	// L side
	auto atten_l = ForwardNode(UPDATE_NODE, FFN_OP);
	// Weight as a matrix in the DIR
	auto attenLWeightInfo = DataInfo(CM_DTYPE);
	attenLWeightInfo.setDims(32, 1); // -2=input embedding dimension, -3=output classes
	auto attenLWeightLevel = DataLevel(&attenLWeightInfo, true);
	auto attenLWeightData = DataNode("attenLWeight1", INT32, INT32, F32, &attenLWeightLevel);
	// Res DIR
	auto attenLInfo = DataInfo(RM_DTYPE);
	attenLInfo.setDims(-1, 1); // -1=N=232965, the number of nodes in the graph, -3=output classes
	auto rootAttenLLevel = DataLevel(&attenLInfo, true);
	auto attenLData = DataNode("attenL", INT32, INT32, F32, &rootAttenLLevel);
	// set dimenions from the new schedule information
	atten_l.addInputData(&resData);
	atten_l.addInputData(&attenLWeightData);
	atten_l.addOutputData(&attenLData);
	trainingLoop.addLoopNode(&atten_l);
	//* Dependencies
	auto inOutAttenLtDepRelationFeat = RelationEdge(&resData, ALL_RELATION, &attenLData, ALL_RELATION);
	auto inOutAttenLDepRelationWeight = RelationEdge(&attenLWeightData, COLS_RELATION, &attenLData, ROWS_RELATION);
	dependencies.push_back(&inOutAttenLtDepRelationFeat);
	dependencies.push_back(&inOutAttenLDepRelationWeight);
	auto inOutAttenLAssociation = RelationEdge(&resData, ROWS_RELATION, &attenLWeightData, COLS_RELATION);
	associations.push_back(&inOutAttenLAssociation);
	// R side
	auto atten_r = ForwardNode(UPDATE_NODE, FFN_OP);
	// Weight as a matrix in the DIR
	auto attenRWeightInfo = DataInfo(CM_DTYPE);
	attenRWeightInfo.setDims(32, 1); // -2=input embedding dimension, -3=output classes
	auto attenRWeightLevel = DataLevel(&attenRWeightInfo, true);
	auto attenRWeightData = DataNode("attenRWeight1", INT32, INT32, F32, &attenRWeightLevel);
	// Res DIR
	auto attenRInfo = DataInfo(RM_DTYPE);
	attenRInfo.setDims(-1, 1); // -1=N=232965, the number of nodes in the graph, -3=output classes
	auto rootAttenRLevel = DataLevel(&attenRInfo, true);
	auto attenRData = DataNode("attenR", INT32, INT32, F32, &rootAttenRLevel);
	// set dimenions from the new schedule information
	atten_r.addInputData(&resData);
	atten_r.addInputData(&attenRWeightData);
	atten_r.addOutputData(&attenRData);
	trainingLoop.addLoopNode(&atten_r);
	//* Dependencies
	auto inOutAttenRtDepRelationFeat = RelationEdge(&resData, ALL_RELATION, &attenRData, ALL_RELATION);
	auto inOutAttenRDepRelationWeight = RelationEdge(&attenRWeightData, COLS_RELATION, &attenRData, ROWS_RELATION);
	dependencies.push_back(&inOutAttenRtDepRelationFeat);
	dependencies.push_back(&inOutAttenRDepRelationWeight);
	auto inOutAttenRAssociation = RelationEdge(&resData, ROWS_RELATION, &attenRWeightData, COLS_RELATION);
	associations.push_back(&inOutAttenRAssociation);

	// Edge aggregation
	auto aggregateEdge = ForwardNode(AGGREGATE_EDGE, AGGREGATE_EDGE_SUM_OP);
	auto aggrEdgeInfo = DataInfo(CSR_STYPE, transformedGraphInfo.getDirected(), true);
	// aggrEdgeInfo.setDims(-4, 1); //-4=E=114M (E = Edges)
	auto rootAggrEdgeLevel = DataLevel(&aggrEdgeInfo, true);
	auto aggrEdgeData = DataNode("attn", INT32, INT32, F32, &rootAggrEdgeLevel);
	aggregateEdge.addInputData(&attenLWeightData);
	aggregateEdge.addInputData(&attenRWeightData);
	aggregateEdge.addInputData(&transformedGraph);
	aggregateEdge.addOutputData(&aggrEdgeData);
	// TODO add optimizations
	trainingLoop.addLoopNode(&aggregateEdge);
	//* Dependencies
	// Dependency relation between the features and the aggregated output
	auto inOutEdgeAggrLRelationFeat = RelationEdge(&attenLWeightData, ALL_RELATION, &aggrEdgeData, ROWS_RELATION);
	auto inOutEdgeAggrRRelationFeat = RelationEdge(&attenRWeightData, ALL_RELATION, &aggrEdgeData, COLS_RELATION);
	auto inOutEdgeAggrRelationGraph = RelationEdge(&transformedGraph, ALL_RELATION, &aggrEdgeData, ALL_RELATION);
	dependencies.push_back(&inOutEdgeAggrLRelationFeat);
	dependencies.push_back(&inOutEdgeAggrRRelationFeat);
	dependencies.push_back(&inOutEdgeAggrRelationGraph);
	auto graphEdgeAggrLAssociation = RelationEdge(&transformedGraph, ROWS_RELATION, &attenLWeightData, ALL_RELATION);
	auto graphEdgeAggrRAssociation = RelationEdge(&transformedGraph, COLS_RELATION, &attenRWeightData, ALL_RELATION);
	associations.push_back(&graphEdgeAggrLAssociation);
	associations.push_back(&graphEdgeAggrRAssociation);

	// Leaky ReLU operation
	auto leakyReluOp = ForwardNode(UPDATE_EDGE, NON_LNR_OP_LEAKY_RELU);
	leakyReluOp.addParam("0.2");
	auto leakyReluInfo = DataInfo(CSR_STYPE, transformedGraphInfo.getDirected(), true);
	// leakyReluInfo.setDims(-4, 1);
	auto rootLeakyReluLevel = DataLevel(&leakyReluInfo, true);
	auto leakyReluData = DataNode("attn", INT32, INT32, F32, &rootLeakyReluLevel);
	leakyReluOp.addInputData(&aggrEdgeData);
	leakyReluOp.addOutputData(&leakyReluData);
	trainingLoop.addLoopNode(&leakyReluOp);
	auto leakyReluOpOnesDependency = RelationEdge(&aggrEdgeData, ALL_RELATION, &leakyReluData, ALL_RELATION);
	dependencies.push_back(&leakyReluOpOnesDependency);

	// Leaky ReLU operation
	auto softmaxOp = ForwardNode(UPDATE_EDGE, NON_LNR_OP_SOFTMAX);
	auto softmaxInfo = DataInfo(CSR_STYPE, transformedGraphInfo.getDirected(), true);
	// leakyReluInfo.setDims(-4, 1);
	auto rootSoftmaxLevel = DataLevel(&softmaxInfo, true);
	auto softmaxData = DataNode("attn", INT32, INT32, F32, &rootSoftmaxLevel);
	softmaxOp.addInputData(&leakyReluData);
	softmaxOp.addOutputData(&softmaxData);
	trainingLoop.addLoopNode(&softmaxOp);
	auto softmaxOpOnesDependency = RelationEdge(&leakyReluData, ALL_RELATION, &softmaxData, ALL_RELATION);
	dependencies.push_back(&softmaxOpOnesDependency);

    // Add aggregate operation
    auto aggregate = ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_OP);
    auto outputInfo = DataInfo(RM_DTYPE);
    outputInfo.setDims(-1, 32); // -1=N=232965, the number of nodes in the graph
    auto rootOutputLevel = DataLevel(&outputInfo, true);
    auto outputData = DataNode("res", INT32, INT32, F32, &rootOutputLevel);
	aggregate.addInputData(&resData);
	aggregate.addInputData(&softmaxData);
    aggregate.addOutputData(&outputData);
	aggregate.addOpt(COARSE_COPT, 2);
    trainingLoop.addLoopNode(&aggregate);
	//* Dependencies
    // Dependency relation between the features and the aggregated output
	auto inOutAggrRelationFeat = RelationEdge(&resData, ALL_RELATION, &outputData, ALL_RELATION);
	// Dependency relation between the graph and the aggregated output
	auto inOutAggrRelationGraph = RelationEdge(&softmaxData, ALL_RELATION, &outputData, ROWS_RELATION);
    dependencies.push_back(&inOutAggrRelationFeat);
    dependencies.push_back(&inOutAggrRelationGraph);

	// ReLU operation
	auto reluOp = ForwardNode(POINTWISE, NON_LNR_OP_RELU);
	auto reluInfo = DataInfo(RM_DTYPE);
	reluInfo.setDims(-1, 32);
	auto rootReluLevel = DataLevel(&reluInfo, true);
	auto reluData = DataNode("res", INT32, INT32, F32, &rootReluLevel);
	reluOp.addInputData(&outputData);
	reluOp.addOutputData(&reluData);
	trainingLoop.addLoopNode(&reluOp);
	auto reluOpOnesDependency = RelationEdge(&outputData, ALL_RELATION, &reluData, ALL_RELATION);
	dependencies.push_back(&reluOpOnesDependency);

	/// 2nd layer
	// Add weight operation
	auto ffn_2 = ForwardNode(UPDATE_NODE, FFN_OP);
	// Weight as a matrix in the DIR
	auto weightInfo_2 = DataInfo(CM_DTYPE);
	weightInfo_2.setDims(32, -3); // -2=input embedding dimension, -3=output classes
	auto weightLevel_2 = DataLevel(&weightInfo_2, true);
	auto weightData_2 = DataNode("weight2", INT32, INT32, F32, &weightLevel_2);
	// Res DIR
	auto resInfo_2 = DataInfo(RM_DTYPE);
	resInfo_2.setDims(-1, -3); // -1=N=232965, the number of nodes in the graph, -3=output classes
	auto rootResLevel_2 = DataLevel(&resInfo_2, true);
	auto resData_2 = DataNode("res", INT32, INT32, F32, &rootResLevel_2);
	// set dimensions from the new schedule information
	weightInfo_2.setDims(32, 41); //
	resInfo_2.setDims(-1, 41); // -1=N=232965, the number of nodes in the graph
	ffn_2.addInputData(&reluData);
	ffn_2.addInputData(&weightData_2);
	ffn_2.addOutputData(&resData_2);
	trainingLoop.addLoopNode(&ffn_2);
	//* Dependencies
	auto inOutWeightDepRelationFeat_2 = RelationEdge(&reluData, ALL_RELATION, &resData_2, ALL_RELATION);
	auto inOutWeightDepRelationWeight_2 = RelationEdge(&weightData_2, COLS_RELATION, &resData_2, ROWS_RELATION);
	dependencies.push_back(&inOutWeightDepRelationFeat_2);
	dependencies.push_back(&inOutWeightDepRelationWeight_2);
	auto inOutWeightAssociation_2 = RelationEdge(&reluData, ROWS_RELATION, &weightData_2, COLS_RELATION);
	associations.push_back(&inOutWeightAssociation_2);

	// Add attention weight operation (L side)
	// L side
	auto atten_l_2 = ForwardNode(UPDATE_NODE, FFN_OP);
	// Weight as a matrix in the DIR
	auto attenLWeightInfo_2 = DataInfo(CM_DTYPE);
	attenLWeightInfo_2.setDims(41, 1); // -2=input embedding dimension, -3=output classes
	auto attenLWeightLevel_2 = DataLevel(&attenLWeightInfo_2, true);
	auto attenLWeightData_2 = DataNode("attenLWeight2", INT32, INT32, F32, &attenLWeightLevel_2);
	// Res DIR
	auto attenLInfo_2 = DataInfo(RM_DTYPE);
	attenLInfo_2.setDims(-1, 1); // -1=N=232965, the number of nodes in the graph, -3=output classes
	auto rootAttenLLevel_2 = DataLevel(&attenLInfo_2, true);
	auto attenLData_2 = DataNode("attenL_2", INT32, INT32, F32, &rootAttenLLevel_2);
	// set dimenions from the new schedule information
	atten_l_2.addInputData(&resData_2);
	atten_l_2.addInputData(&attenLWeightData_2);
	atten_l_2.addOutputData(&attenLData_2);
	trainingLoop.addLoopNode(&atten_l_2);
	//* Dependencies
	auto inOutAttenLtDepRelationFeat_2 = RelationEdge(&resData_2, ALL_RELATION, &attenLData_2, ALL_RELATION);
	auto inOutAttenLDepRelationWeight_2 = RelationEdge(&attenLWeightData_2, COLS_RELATION, &attenLData_2, ROWS_RELATION);
	dependencies.push_back(&inOutAttenLtDepRelationFeat_2);
	dependencies.push_back(&inOutAttenLDepRelationWeight_2);
	auto inOutAttenLAssociation_2 = RelationEdge(&resData_2, ROWS_RELATION, &attenLWeightData_2, COLS_RELATION);
	associations.push_back(&inOutAttenLAssociation_2);
	// R side
	auto atten_r_2 = ForwardNode(UPDATE_NODE, FFN_OP);
	// Weight as a matrix in the DIR
	auto attenRWeightInfo_2 = DataInfo(CM_DTYPE);
	attenRWeightInfo_2.setDims(41, 1); // -2=input embedding dimension, -3=output classes
	auto attenRWeightLevel_2 = DataLevel(&attenRWeightInfo_2, true);
	auto attenRWeightData_2 = DataNode("attenRWeight2", INT32, INT32, F32, &attenRWeightLevel_2);
	// Res DIR
	auto attenRInfo_2 = DataInfo(RM_DTYPE);
	attenRInfo_2.setDims(-1, 1); // -1=N=232965, the number of nodes in the graph, -3=output classes
	auto rootAttenRLevel_2 = DataLevel(&attenRInfo_2, true);
	auto attenRData_2 = DataNode("attenR_2", INT32, INT32, F32, &rootAttenRLevel_2);
	// set dimenions from the new schedule information
	atten_r_2.addInputData(&resData_2);
	atten_r_2.addInputData(&attenRWeightData_2);
	atten_r_2.addOutputData(&attenRData_2);
	trainingLoop.addLoopNode(&atten_r_2);
	//* Dependencies
	auto inOutAttenRtDepRelationFeat_2 = RelationEdge(&resData_2, ALL_RELATION, &attenRData_2, ALL_RELATION);
	auto inOutAttenRDepRelationWeight_2 = RelationEdge(&attenRWeightData_2, COLS_RELATION, &attenRData_2, ROWS_RELATION);
	dependencies.push_back(&inOutAttenRtDepRelationFeat_2);
	dependencies.push_back(&inOutAttenRDepRelationWeight_2);
	auto inOutAttenRAssociation_2 = RelationEdge(&resData_2, ROWS_RELATION, &attenRWeightData_2, COLS_RELATION);
	associations.push_back(&inOutAttenRAssociation_2);

	// Edge aggregation
	auto aggregateEdge_2 = ForwardNode(AGGREGATE_EDGE, AGGREGATE_MUL_SUM_OP);
	auto aggrEdgeInfo_2 = DataInfo(CSR_STYPE, transformedGraphInfo.getDirected(), true);
	// aggrEdgeInfo.setDims(-4, 1); //-4=E=114M (E = Edges)
	auto rootAggrEdgeLevel_2 = DataLevel(&aggrEdgeInfo_2, true);
	auto aggrEdgeData_2 = DataNode("attn", INT32, INT32, F32, &rootAggrEdgeLevel_2);
	aggregateEdge_2.addInputData(&attenLWeightData_2);
	aggregateEdge_2.addInputData(&attenRWeightData_2);
	aggregateEdge_2.addInputData(&transformedGraph);
	aggregateEdge_2.addOutputData(&aggrEdgeData_2);
	// TODO add optimizations
	trainingLoop.addLoopNode(&aggregateEdge_2);
	//* Dependencies
	// Dependency relation between the features and the aggregated output
	auto inOutEdgeAggrLRelationFeat_2 = RelationEdge(&attenLWeightData_2, ALL_RELATION, &aggrEdgeData_2, ROWS_RELATION);
	auto inOutEdgeAggrRRelationFeat_2 = RelationEdge(&attenRWeightData_2, ALL_RELATION, &aggrEdgeData_2, COLS_RELATION);
	auto inOutEdgeAggrRelationGraph_2 = RelationEdge(&transformedGraph, ALL_RELATION, &aggrEdgeData_2, ALL_RELATION);
	dependencies.push_back(&inOutEdgeAggrLRelationFeat_2);
	dependencies.push_back(&inOutEdgeAggrRRelationFeat_2);
	dependencies.push_back(&inOutEdgeAggrRelationGraph_2);
	auto graphEdgeAggrLAssociation_2 = RelationEdge(&transformedGraph, ROWS_RELATION, &attenLWeightData_2, ALL_RELATION);
	auto graphEdgeAggrRAssociation_2 = RelationEdge(&transformedGraph, COLS_RELATION, &attenRWeightData_2, ALL_RELATION);
	associations.push_back(&graphEdgeAggrLAssociation_2);
	associations.push_back(&graphEdgeAggrRAssociation_2);

	// Leaky ReLU operation
	auto leakyReluOp_2 = ForwardNode(UPDATE_EDGE, NON_LNR_OP_LEAKY_RELU);
	leakyReluOp_2.addParam("0.2");
	auto leakyReluInfo_2 = DataInfo(CSR_STYPE, transformedGraphInfo.getDirected(), true);
	// leakyReluInfo.setDims(-4, 1);
	auto rootLeakyReluLevel_2 = DataLevel(&leakyReluInfo_2, true);
	auto leakyReluData_2 = DataNode("attn", INT32, INT32, F32, &rootLeakyReluLevel_2);
	leakyReluOp_2.addInputData(&aggrEdgeData_2);
	leakyReluOp_2.addOutputData(&leakyReluData_2);
	trainingLoop.addLoopNode(&leakyReluOp_2);
	auto leakyReluOpOnesDependency_2 = RelationEdge(&aggrEdgeData_2, ALL_RELATION, &leakyReluData_2, ALL_RELATION);
	dependencies.push_back(&leakyReluOpOnesDependency_2);

	// Leaky ReLU operation
	auto softmaxOp_2 = ForwardNode(UPDATE_EDGE, NON_LNR_OP_SOFTMAX);
	auto softmaxInfo_2 = DataInfo(CSR_STYPE, transformedGraphInfo.getDirected(), true);
	// leakyReluInfo.setDims(-4, 1);
	auto rootSoftmaxLevel_2 = DataLevel(&softmaxInfo_2, true);
	auto softmaxData_2 = DataNode("attn", INT32, INT32, F32, &rootSoftmaxLevel_2);
	softmaxOp_2.addInputData(&leakyReluData_2);
	softmaxOp_2.addOutputData(&softmaxData_2);
	trainingLoop.addLoopNode(&softmaxOp_2);
	auto softmaxOpOnesDependency_2 = RelationEdge(&leakyReluData_2, ALL_RELATION, &softmaxData_2, ALL_RELATION);
	dependencies.push_back(&softmaxOpOnesDependency_2);

    // Add aggregate operation
    auto aggregate_2 = ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_OP);
    auto outputInfo_2 = DataInfo(RM_DTYPE);
    outputInfo_2.setDims(-1, 32); // -1=N=232965, the number of nodes in the graph
    auto rootOutputLevel_2 = DataLevel(&outputInfo_2, true);
    auto outputData_2 = DataNode("res", INT32, INT32, F32, &rootOutputLevel_2);
	aggregate_2.addInputData(&resData_2);
	aggregate_2.addInputData(&softmaxData_2);
    aggregate_2.addOutputData(&outputData_2);
	aggregate_2.addOpt(COARSE_COPT, 2);
    trainingLoop.addLoopNode(&aggregate_2);
	//* Dependencies
    // Dependency relation between the features and the aggregated output
	auto inOutAggrRelationFeat_2 = RelationEdge(&resData_2, ALL_RELATION, &outputData_2, ALL_RELATION);
	// Dependency relation between the graph and the aggregated output
	auto inOutAggrRelationGraph_2 = RelationEdge(&softmaxData_2, ALL_RELATION, &outputData_2, ROWS_RELATION);
    dependencies.push_back(&inOutAggrRelationFeat_2);
    dependencies.push_back(&inOutAggrRelationGraph_2);

	// ReLU operation
	auto logSoftmaxOp = ForwardNode(POINTWISE, NON_LNR_OP_LOG_SOFTMAX);
	auto logSoftmaxInfo = DataInfo(RM_DTYPE);
	logSoftmaxInfo.setDims(-1, 41);
	auto rootLogSoftmaxLevel = DataLevel(&logSoftmaxInfo, true);
	auto logSoftmaxData = DataNode("res", INT32, INT32, F32, &rootLogSoftmaxLevel);
	logSoftmaxOp.addInputData(&outputData_2);
	logSoftmaxOp.addOutputData(&logSoftmaxData);
	trainingLoop.addLoopNode(&logSoftmaxOp);
	auto logSoftmaxOpOnesDependency = RelationEdge(&outputData_2, ALL_RELATION, &logSoftmaxData, ALL_RELATION);
	dependencies.push_back(&logSoftmaxOpOnesDependency);

    // The entire program
    program.push_back(&loadDataset);
	program.push_back(&trainingLoop);

	auto ctx = new GALAContext(GPU_DEVICE, SINGLE_NODE_SINGLE);
	std::string outputPath = "../test-codegen/";
	auto genCode = CUDAGenerator(ctx, outputPath);
	GALATransformations::complexityOperatorReordering(program, dependencies, associations, transforms);
	GALATransformations::trainingInvariantCodeMotion(program, dependencies, associations, transforms);
	genCode.writeCode(program, dependencies, associations, transforms);

    // Should be enough for now
	std::cout << "Works!" << std::endl;
}