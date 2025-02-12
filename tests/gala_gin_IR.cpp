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
    auto featData = DataNode("t_iden", INT32, INT32, F32, &rootFeatLevel);

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

    // Add aggregate operation
    auto aggregate = ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_OP);
    auto outputInfo = DataInfo(RM_DTYPE);
    outputInfo.setDims(-1, 605); // -1=N=232965, the number of nodes in the graph
    auto rootOutputLevel = DataLevel(&outputInfo, true);
    auto outputData = DataNode("res_n", INT32, INT32, F32, &rootOutputLevel);
	aggregate.addInputData(&featData);
	aggregate.addInputData(&transformedGraph);
    aggregate.addOutputData(&outputData);
	aggregate.addOpt(COARSE_COPT, 2);
    trainingLoop.addLoopNode(&aggregate);
	//* Dependencies
    // Dependency relation between the features and the aggregated output
	auto inOutAggrRelationFeat = RelationEdge(&featData, ALL_RELATION, &outputData, ALL_RELATION);
	// Dependency relation between the graph and the aggregated output
	auto inOutAggrRelationGraph = RelationEdge(&transformedGraph, ALL_RELATION, &outputData, ROWS_RELATION);
    dependencies.push_back(&inOutAggrRelationFeat);
    dependencies.push_back(&inOutAggrRelationGraph);

    // Scalar multiply res
	auto scalarEps = ForwardNode(POINTWISE, SCALAR_ADD_EPS_MULTIPLY_OP);
	scalarEps.addParam("1");
	auto scalarInfo = DataInfo(RM_DTYPE);
	scalarInfo.setDims(-1, 605);
	auto rootScalarEpsLevel = DataLevel(&scalarInfo, true);
	auto scalarEpsData = DataNode("res_e", INT32, INT32, F32, &rootScalarEpsLevel);
	scalarEps.addInputData(&featData);
	scalarEps.addOutputData(&scalarEpsData);
	trainingLoop.addLoopNode(&scalarEps);
	auto scalarEpsDependency = RelationEdge(&featData, ALL_RELATION, &scalarEpsData, ALL_RELATION);
	dependencies.push_back(&scalarEpsDependency);

    // Add epsilon mult and scalar mults
	auto normFeat = ForwardNode(UPDATE_NODE, ADD_OP);
	auto normFeatInfo = DataInfo(RM_DTYPE);
	normFeatInfo.setDims(-1, 605);
	auto rootNormFeatLevel = DataLevel(&normFeatInfo, true);
	auto normFeatData = DataNode("res", INT32, INT32, F32, &rootNormFeatLevel);
	normFeat.addInputData(&scalarEpsData);
	normFeat.addInputData(&outputData);
	normFeat.addOutputData(&normFeatData);
	trainingLoop.addLoopNode(&normFeat);
	auto normFeatNormDependency = RelationEdge(&scalarEpsData, ALL_RELATION, &normFeatData, ALL_RELATION);
	dependencies.push_back(&normFeatNormDependency);
	auto normFeatFeatDependency = RelationEdge(&outputData, ALL_RELATION, &normFeatData, ALL_RELATION);
	dependencies.push_back(&normFeatFeatDependency);
	auto normFeatNormFeatAssociation = RelationEdge(&scalarEpsData, ALL_RELATION, &outputData, ALL_RELATION);
	associations.push_back(&normFeatNormFeatAssociation);

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
    ffn.addInputData(&normFeatData);
    ffn.addInputData(&weightData);
    ffn.addOutputData(&resData);
    trainingLoop.addLoopNode(&ffn);
	//* Dependencies
    auto inOutWeightDepRelationFeat = RelationEdge(&normFeatData, ALL_RELATION, &resData, ALL_RELATION);
    auto inOutWeightDepRelationWeight = RelationEdge(&weightData, COLS_RELATION, &resData, ROWS_RELATION);
    dependencies.push_back(&inOutWeightDepRelationFeat);
    dependencies.push_back(&inOutWeightDepRelationWeight);
    auto inOutWeightAssociation = RelationEdge(&normFeatData, ROWS_RELATION, &weightData, COLS_RELATION);
    associations.push_back(&inOutWeightAssociation);

	// ReLU operation
	auto reluOp = ForwardNode(POINTWISE, NON_LNR_OP_RELU);
	auto reluInfo = DataInfo(RM_DTYPE);
	reluInfo.setDims(-1, 32);
	auto rootReluLevel = DataLevel(&reluInfo, true);
	auto reluData = DataNode("res", INT32, INT32, F32, &rootReluLevel);
	reluOp.addInputData(&resData);
	reluOp.addOutputData(&reluData);
	trainingLoop.addLoopNode(&reluOp);
	auto reluOpOnesDependency = RelationEdge(&resData, ALL_RELATION, &reluData, ALL_RELATION);
	dependencies.push_back(&reluOpOnesDependency);

	/// 2nd layer
	// Add aggregate operation
    auto aggregate_2 = ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_OP);
    auto outputInfo_2 = DataInfo(RM_DTYPE);
    outputInfo.setDims(-1, 32); // -1=N=232965, the number of nodes in the graph
    auto rootOutputLevel_2 = DataLevel(&outputInfo_2, true);
    auto outputData_2 = DataNode("res_n", INT32, INT32, F32, &rootOutputLevel_2);
	aggregate_2.addInputData(&reluData);
	aggregate_2.addInputData(&transformedGraph);
    aggregate_2.addOutputData(&outputData_2);
	aggregate_2.addOpt(COARSE_COPT, 2);
    trainingLoop.addLoopNode(&aggregate_2);
	//* Dependencies
    // Dependency relation between the features and the aggregated output
	auto inOutAggrRelationFeat_2 = RelationEdge(&reluData, ALL_RELATION, &outputData_2, ALL_RELATION);
	// Dependency relation between the graph and the aggregated output
	auto inOutAggrRelationGraph_2 = RelationEdge(&transformedGraph, ALL_RELATION, &outputData_2, ROWS_RELATION);
    dependencies.push_back(&inOutAggrRelationFeat_2);
    dependencies.push_back(&inOutAggrRelationGraph_2);

	// Scalar multiply res
	auto scalarEps_2 = ForwardNode(POINTWISE, SCALAR_ADD_EPS_MULTIPLY_OP);
	scalarEps_2.addParam("1");
	auto scalarInfo_2 = DataInfo(RM_DTYPE);
	scalarInfo.setDims(-1, 32);
	auto rootScalarEpsLevel_2 = DataLevel(&scalarInfo_2, true);
	auto scalarEpsData_2 = DataNode("res_e", INT32, INT32, F32, &rootScalarEpsLevel_2);
	scalarEps_2.addInputData(&reluData);
	scalarEps_2.addOutputData(&scalarEpsData_2);
	trainingLoop.addLoopNode(&scalarEps_2);
	auto scalarEpsDependency_2 = RelationEdge(&reluData, ALL_RELATION, &scalarEpsData_2, ALL_RELATION);
	dependencies.push_back(&scalarEpsDependency_2);

	// Add epsilon mult and scalar mults
	auto normFeat_2 = ForwardNode(UPDATE_NODE, ADD_OP);
	auto normFeatInfo_2 = DataInfo(RM_DTYPE);
	normFeatInfo_2.setDims(-1, 32);
	auto rootNormFeatLevel_2 = DataLevel(&normFeatInfo_2, true);
	auto normFeatData_2 = DataNode("res", INT32, INT32, F32, &rootNormFeatLevel_2);
	normFeat_2.addInputData(&scalarEpsData_2);
	normFeat_2.addInputData(&outputData_2);
	normFeat_2.addOutputData(&normFeatData_2);
	trainingLoop.addLoopNode(&normFeat_2);
	auto normFeatNormDependency_2 = RelationEdge(&scalarEpsData_2, ALL_RELATION, &normFeatData_2, ALL_RELATION);
	dependencies.push_back(&normFeatNormDependency_2);
	auto normFeatFeatDependency_2 = RelationEdge(&outputData_2, ALL_RELATION, &normFeatData_2, ALL_RELATION);
	dependencies.push_back(&normFeatFeatDependency_2);
	auto normFeatNormFeatAssociation_2 = RelationEdge(&scalarEpsData_2, ALL_RELATION, &outputData_2, ALL_RELATION);
	associations.push_back(&normFeatNormFeatAssociation_2);

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
	// set dimenions from the new schedule information
	weightInfo_2.setDims(32, 41); //
	resInfo_2.setDims(-1, 41); // -1=N=232965, the number of nodes in the graph
    ffn_2.addInputData(&normFeatData_2);
    ffn_2.addInputData(&weightData_2);
    ffn_2.addOutputData(&resData_2);
    trainingLoop.addLoopNode(&ffn_2);
	//* Dependencies
    auto inOutWeightDepRelationFeat_2 = RelationEdge(&normFeatData_2, ALL_RELATION, &resData_2, ALL_RELATION);
    auto inOutWeightDepRelationWeight_2 = RelationEdge(&weightData_2, COLS_RELATION, &resData_2, ROWS_RELATION);
    dependencies.push_back(&inOutWeightDepRelationFeat_2);
    dependencies.push_back(&inOutWeightDepRelationWeight_2);
    auto inOutWeightAssociation_2 = RelationEdge(&normFeatData_2, ROWS_RELATION, &weightData_2, COLS_RELATION);
    associations.push_back(&inOutWeightAssociation_2);

	// ReLU operation
	auto logSoftmaxOp = ForwardNode(POINTWISE, NON_LNR_OP_LOG_SOFTMAX);
	auto logSoftmaxInfo = DataInfo(RM_DTYPE);
	logSoftmaxInfo.setDims(-1, 41);
	auto rootLogSoftmaxLevel = DataLevel(&logSoftmaxInfo, true);
	auto logSoftmaxData = DataNode("res", INT32, INT32, F32, &rootLogSoftmaxLevel);
	logSoftmaxOp.addInputData(&resData_2);
	logSoftmaxOp.addOutputData(&logSoftmaxData);
	trainingLoop.addLoopNode(&logSoftmaxOp);
	auto logSoftmaxOpOnesDependency = RelationEdge(&resData_2, ALL_RELATION, &logSoftmaxData, ALL_RELATION);
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