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
#include "../src/ir/frontend_metadata.h"
#include "../src/codegen/cuda.h"
#include "../src/middle-end/middle-end.h"

// Code generator
//#include "../src/codegen//gala/codegen/cuda.h"

// Matrix classes
//#include "../src/utils/mtx_io.h"
#include "../src/formats/dense_matrix.h"
#include "../src/formats/csrc_matrix.h"

// Frontend
#include "../src/frontend/context.h"

// #pragma once
void generate_ir();

extern FILE* yyin;
extern int yyparse();
ModelConfig m1 = ModelConfig();
std::vector<CIRNode*> program;
std::vector<RelationEdge*> dependencies;
std::vector<RelationEdge*> associations;
std::vector<TransformEdge*> transforms;

std::vector<CIRNode*>* programP = nullptr;
std::vector<RelationEdge*>* dependenciesP = nullptr;
std::vector<RelationEdge*>* associationsP = nullptr;
std::vector<TransformEdge*>* transformsP = nullptr;

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
	const char* fileName= "../tests/input/gcn.txt";
	// const char* fileName= "/home/damitha/gala-lang/GNN-Acceleration-Language/tests/input/gcn.txt";

	// std::vector<CIRNode*> newProgram;
	// program = &newProgram;
	// std::vector<RelationEdge*> newDependencies;
	// dependencies = &newDependencies;
	// std::vector<RelationEdge*> newAssociations;
	// associations = &newAssociations;
	// std::vector<TransformEdge*> newTransforms;
	// transforms = &newTransforms;

	FILE *myfile = fopen(fileName, "r");
	if (!myfile) {
		std::cout << "Invalid File" << std::endl;
		return -1;
	}
	yyin = myfile;
	yyparse();
	fclose(myfile);

	cout << " ---------------- printing model config ----------------------\n";
	cout << m1.to_string() << '\n';
	cout << "---------------------------------------------------------------\n";

	generate_ir();

	cout << " --------     checking generated ir output ------------ \n";
	cout << "PROGRAM (CIR Nodes): " << program.size() << '\n';
	programP = &program;
	dependenciesP = &dependencies;
	associationsP = &associations;
	transformsP = &transforms;

	for (int i = 0; i < program.size(); i++){

		/* ComputeNode* brruv = dynamic_cast<ComputeNode*>(program[i]); */
		/* cout << "     program node " << i << " with op and opType " << brruv->getOp() << ' ' << brruv->getOpType() << '\n'; */
		cout << "        program node " << i << "\n";
	}
	cout << "DEPENDENCIES " << dependencies.size() << '\n';
	for (int i = 0; i < dependencies.size(); i++){
		cout << "     dependency edge " << i << " with nodes " << dependencies[i]->getNode1()->getName() << ", " << dependencies[i]->getNode2()->getName() << '\n';
	}
	cout << "ASSOCIATIONS " << associations.size() << '\n';
	for (int i = 0; i < associations.size(); i++){
		cout << "     associations edge " << i << " with nodes " << associations[i]->getNode1()->getName() << ", " << associations[i]->getNode2()->getName() << '\n';
	}
	cout << "TRANSFORMS " << transforms.size() << '\n';
	for (int i = 0; i < transforms.size(); i++){
		cout << "     transform edge " << i << " with nodes " << transforms[i]->getNode1()->getName() << ", " << transforms[i]->getNode2()->getName() << '\n';
	}

 //    auto loadDataset = ForwardNode(POINTWISE, LOAD_OP);
 //    loadDataset.addParam("/shared/damitha2/gala_npy/RedditDataset/");
 //
 //    // Graph
 //    auto graphInfo = DataInfo(CSR_STYPE, true, true);
 //    auto rootGraphLevel = DataLevel(&graphInfo, true);
 //    auto graphData = DataNode("adj0", INT32, INT32, F32, &rootGraphLevel);
 //    // Feat
 //    auto featInfo = DataInfo(RM_DTYPE);
 //    featInfo.setDims(-1, -2);
 //    auto rootFeatLevel = DataLevel(&featInfo, true);
 //    auto featData = DataNode("t_iden", INT32, INT32, F32, &rootFeatLevel);
 //
	// // Association between graph and features
	// auto graphFeatAssociation = RelationEdge(&graphData, ALL_RELATION, &featData, ROWS_RELATION);
	// associations->push_back(&graphFeatAssociation);
	// loadDataset.addOutputData(&featData);
	// loadDataset.addOutputData(&graphData);
 //
	// auto originalRootGraphLevel = graphData.getData(); // Returns pointer
	// auto originalGraphInfo = originalRootGraphLevel->next(); // Returns pointer
	// auto transformedGraphInfo = DataInfo(CSR_STYPE, false, false);
	// transformedGraphInfo.addOpt(COL_TILE_DOPT, "65000");
	// auto transformedTileGraphLevel = DataLevel(&transformedGraphInfo, false);
	// auto transformedRootGraphLevel = DataLevel(&transformedTileGraphLevel, true);
	// auto transformedGraph = DataNode("graph_tile", graphData.getIType(), graphData.getNType(), graphData.getVType(), &transformedRootGraphLevel);
	// // Association between graph and features
	// auto trgrapgFeatAssociation = RelationEdge(&transformedGraph, ALL_RELATION, &featData, ROWS_RELATION);
	// associations->push_back(&graphFeatAssociation);
	// auto tileTransformation = TransformData(COL_TILE_DOPT);
	// tileTransformation.addParam("65000");
	// auto graphTrgraph = TransformEdge(&graphData, &transformedGraph);
	// graphTrgraph.addTransformation(&tileTransformation);
	// transforms->push_back(&graphTrgraph);
 //
	// featInfo.setDims(-1, 602);
 //
	// auto trainingLoop = TrainingLoopNode(100, CROSS_ENTROPY, ADAM, 0, 5);
 //
	// /// Degree op
	// // Ones
	// // NOTE: In the front-end code there is no need for a user to get the degrees() by getting ones, and doing an
	// // aggregation. This is just how it's done under the hood.
	// auto onesTensorOp = ForwardNode(POINTWISE, ONES_OP);
	// auto onesInfo = DataInfo(RM_DTYPE);
	// onesInfo.setDims(-1, 1);
	// auto rootOnesLevel = DataLevel(&onesInfo, true);
	// auto onesData = DataNode("ones", INT32, INT32, F32, &rootOnesLevel);
	// onesTensorOp.addOutputData(&onesData);
	// trainingLoop.addLoopNode(&onesTensorOp);
	// //* Dependencies
	// auto onesTensorGraphAssociation = RelationEdge(&transformedGraph, ALL_RELATION, &onesData, ROWS_RELATION);
	// associations->push_back(&onesTensorGraphAssociation);
 //
	// // The actual degrees calculation
	// auto degreesOp = ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_DIRECT);
	// auto degreesInfo = DataInfo(RM_DTYPE);
	// degreesInfo.setDims(-1, 1);
	// auto rootDegreesLevel = DataLevel(&degreesInfo, true);
	// auto degreesData = DataNode("degrees", INT32, INT32, F32, &rootDegreesLevel);
	// degreesOp.addInputData(&onesData);
	// degreesOp.addInputData(&transformedGraph);
	// degreesOp.addOutputData(&degreesData);
	// degreesOp.addOpt(COARSE_COPT, 2);
	// trainingLoop.addLoopNode(&degreesOp);
	// auto degreesOpOnesDependency = RelationEdge(&onesData, ALL_RELATION, &degreesData, ALL_RELATION);
	// dependencies->push_back(&degreesOpOnesDependency);
	// auto degreesOpGraphDependency = RelationEdge(&transformedGraph, ALL_RELATION, &degreesData, ROWS_RELATION);
	// dependencies->push_back(&degreesOpGraphDependency);
 //
	// // Normalization operation (get power -1/2)
	// auto powerOp = ForwardNode(POINTWISE, POWER_OP);
	// powerOp.addParam("-0.5");
	// auto normInfo = DataInfo(RM_DTYPE);
	// normInfo.setDims(-1, 1);
	// auto rootNormLevel = DataLevel(&normInfo, true);
	// auto normData = DataNode("norm", INT32, INT32, F32, &rootNormLevel);
	// powerOp.addInputData(&degreesData);
	// powerOp.addOutputData(&normData);
	// trainingLoop.addLoopNode(&powerOp);
	// auto powerOpNormDependency = RelationEdge(&degreesData, ALL_RELATION, &normData, ALL_RELATION);
	// dependencies->push_back(&powerOpNormDependency);
 //
	// // 1st normalization calculation
	// auto normFeat1 = ForwardNode(UPDATE_NODE, ROW_BROADCAST_OP);
	// auto normFeat1Info = DataInfo(RM_DTYPE);
	// normFeat1Info.setDims(-1, 602);
	// auto rootNormFeat1Level = DataLevel(&normFeat1Info, true);
	// auto normFeat1Data = DataNode("res", INT32, INT32, F32, &rootNormFeat1Level);
	// normFeat1.addInputData(&normData);
	// normFeat1.addInputData(&featData);
	// normFeat1.addOutputData(&normFeat1Data);
	// trainingLoop.addLoopNode(&normFeat1);
	// auto normFeat1NormDependency = RelationEdge(&normData, ALL_RELATION, &normFeat1Data, ROWS_RELATION);
	// dependencies->push_back(&normFeat1NormDependency);
	// auto normFeat1FeatDependency = RelationEdge(&featData, ALL_RELATION, &normFeat1Data, ALL_RELATION);
	// dependencies->push_back(&normFeat1FeatDependency);
	// auto normFeat1NormFeatAssociation = RelationEdge(&normData, ALL_RELATION, &featData, ROWS_RELATION);
	// associations->push_back(&normFeat1NormFeatAssociation);
 //
 //    // Add aggregate operation
 //    auto aggregate = ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_OP);
 //    auto outputInfo = DataInfo(RM_DTYPE);
 //    outputInfo.setDims(-1, 602); // -1=N=232965, the number of nodes in the graph
 //    auto rootOutputLevel = DataLevel(&outputInfo, true);
 //    auto outputData = DataNode("res", INT32, INT32, F32, &rootOutputLevel);
	// aggregate.addInputData(&normFeat1Data);
	// aggregate.addInputData(&transformedGraph);
 //    aggregate.addOutputData(&outputData);
	// aggregate.addOpt(COARSE_COPT, 2);
 //    trainingLoop.addLoopNode(&aggregate);
	// //* Dependencies
 //    // Dependency relation between the features and the aggregated output
	// auto inOutAggrRelationFeat = RelationEdge(&normFeat1Data, ALL_RELATION, &outputData, ALL_RELATION);
	// // Dependency relation between the graph and the aggregated output
	// auto inOutAggrRelationGraph = RelationEdge(&transformedGraph, ALL_RELATION, &outputData, ROWS_RELATION);
 //    dependencies->push_back(&inOutAggrRelationFeat);
 //    dependencies->push_back(&inOutAggrRelationGraph);
 //
 //    // Add weight operation
 //    auto ffn = ForwardNode(UPDATE_NODE, FFN_OP);
 //    // Weight as a matrix in the DIR
 //    auto weightInfo = DataInfo(CM_DTYPE);
 //    weightInfo.setDims(-2, 32); // -2=input embedding dimension, -3=output classes
 //    auto weightLevel = DataLevel(&weightInfo, true);
 //    auto weightData = DataNode("weight1", INT32, INT32, F32, &weightLevel);
 //    // Res DIR
 //    auto resInfo = DataInfo(RM_DTYPE);
 //    resInfo.setDims(-1, 32); // -1=N=232965, the number of nodes in the graph, -3=output classes
	// auto rootResLevel = DataLevel(&resInfo, true);
	// auto resData = DataNode("res", INT32, INT32, F32, &rootResLevel);
	// // set dimenions from the new schedule information
	// weightInfo.setDims(602, 32); //
	// resInfo.setDims(-1, 32); // -1=N=232965, the number of nodes in the graph
 //    ffn.addInputData(&outputData);
 //    ffn.addInputData(&weightData);
 //    ffn.addOutputData(&resData);
 //    trainingLoop.addLoopNode(&ffn);
	// //* Dependencies
 //    auto inOutWeightDepRelationFeat = RelationEdge(&outputData, ALL_RELATION, &resData, ALL_RELATION);
 //    auto inOutWeightDepRelationWeight = RelationEdge(&weightData, COLS_RELATION, &resData, ROWS_RELATION);
 //    dependencies->push_back(&inOutWeightDepRelationFeat);
 //    dependencies->push_back(&inOutWeightDepRelationWeight);
 //    auto inOutWeightAssociation = RelationEdge(&outputData, ROWS_RELATION, &weightData, COLS_RELATION);
 //    associations->push_back(&inOutWeightAssociation);
 //
	// // 2nd normalization calculation
	// auto normFeat2 = ForwardNode(UPDATE_NODE, ROW_BROADCAST_OP);
	// auto normFeat2Info = DataInfo(RM_DTYPE);
	// normFeat2Info.setDims(-1, 32);
	// auto rootNormFeat2Level = DataLevel(&normFeat2Info, true);
	// auto normFeat2Data = DataNode("res", INT32, INT32, F32, &rootNormFeat2Level);
	// normFeat2.addInputData(&normData);
	// normFeat2.addInputData(&resData);
	// normFeat2.addOutputData(&normFeat2Data);
	// trainingLoop.addLoopNode(&normFeat2);
	// auto normFeat2NormDependency = RelationEdge(&normData, ALL_RELATION, &normFeat2Data, ROWS_RELATION);
	// dependencies->push_back(&normFeat2NormDependency);
	// auto normFeat2FeatDependency = RelationEdge(&resData, ALL_RELATION, &normFeat2Data, ALL_RELATION);
	// dependencies->push_back(&normFeat2FeatDependency);
	// auto normFeat2NormFeatAssociation = RelationEdge(&normData, ALL_RELATION, &resData, ROWS_RELATION);
	// associations->push_back(&normFeat2NormFeatAssociation);
 //
	// // ReLU operation
	// auto reluOp = ForwardNode(POINTWISE, NON_LNR_OP_RELU);
	// auto reluInfo = DataInfo(RM_DTYPE);
	// reluInfo.setDims(-1, 32);
	// auto rootReluLevel = DataLevel(&reluInfo, true);
	// auto reluData = DataNode("res", INT32, INT32, F32, &rootReluLevel);
	// reluOp.addInputData(&normFeat2Data);
	// reluOp.addOutputData(&reluData);
	// trainingLoop.addLoopNode(&reluOp);
	// auto reluOpOnesDependency = RelationEdge(&normFeat2Data, ALL_RELATION, &reluData, ALL_RELATION);
	// dependencies->push_back(&reluOpOnesDependency);
 //
	// /// 2nd layer
	// // 1st normalization calculation
	// auto normFeat1_2 = ForwardNode(UPDATE_NODE, ROW_BROADCAST_OP);
	// auto normFeat1Info_2 = DataInfo(RM_DTYPE);
	// normFeat1Info_2.setDims(-1, 32);
	// auto rootNormFeat1Level_2 = DataLevel(&normFeat1Info_2, true);
	// auto normFeat1Data_2 = DataNode("res", INT32, INT32, F32, &rootNormFeat1Level_2);
	// normFeat1_2.addInputData(&normData);
	// normFeat1_2.addInputData(&reluData);
	// normFeat1_2.addOutputData(&normFeat1Data_2);
	// trainingLoop.addLoopNode(&normFeat1_2);
	// auto normFeat1NormDependency_2 = RelationEdge(&normData, ALL_RELATION, &normFeat1Data_2, ROWS_RELATION);
	// dependencies->push_back(&normFeat1NormDependency_2);
	// auto normFeat1FeatDependency_2 = RelationEdge(&reluData, ALL_RELATION, &normFeat1Data_2, ALL_RELATION);
	// dependencies->push_back(&normFeat1FeatDependency_2);
	// auto normFeat1NormFeatAssociation_2 = RelationEdge(&normData, ALL_RELATION, &reluData, ROWS_RELATION);
	// associations->push_back(&normFeat1NormFeatAssociation_2);
 //
 //    // Add aggregate operation
 //    auto aggregate_2 = ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_OP);
 //    auto outputInfo_2 = DataInfo(RM_DTYPE);
 //    outputInfo.setDims(-1, 32); // -1=N=232965, the number of nodes in the graph
 //    auto rootOutputLevel_2 = DataLevel(&outputInfo_2, true);
 //    auto outputData_2 = DataNode("res", INT32, INT32, F32, &rootOutputLevel_2);
	// aggregate_2.addInputData(&normFeat1Data_2);
	// aggregate_2.addInputData(&transformedGraph);
 //    aggregate_2.addOutputData(&outputData_2);
	// aggregate_2.addOpt(COARSE_COPT, 2);
 //    trainingLoop.addLoopNode(&aggregate_2);
	// //* Dependencies
 //    // Dependency relation between the features and the aggregated output
	// auto inOutAggrRelationFeat_2 = RelationEdge(&normFeat1Data_2, ALL_RELATION, &outputData_2, ALL_RELATION);
	// // Dependency relation between the graph and the aggregated output
	// auto inOutAggrRelationGraph_2 = RelationEdge(&transformedGraph, ALL_RELATION, &outputData_2, ROWS_RELATION);
 //    dependencies->push_back(&inOutAggrRelationFeat_2);
 //    dependencies->push_back(&inOutAggrRelationGraph_2);
 //
 //    // Add weight operation
 //    auto ffn_2 = ForwardNode(UPDATE_NODE, FFN_OP);
 //    // Weight as a matrix in the DIR
 //    auto weightInfo_2 = DataInfo(CM_DTYPE);
 //    weightInfo_2.setDims(32, -3); // -2=input embedding dimension, -3=output classes
 //    auto weightLevel_2 = DataLevel(&weightInfo_2, true);
 //    auto weightData_2 = DataNode("weight2", INT32, INT32, F32, &weightLevel_2);
 //    // Res DIR
 //    auto resInfo_2 = DataInfo(RM_DTYPE);
 //    resInfo_2.setDims(-1, -3); // -1=N=232965, the number of nodes in the graph, -3=output classes
	// auto rootResLevel_2 = DataLevel(&resInfo_2, true);
	// auto resData_2 = DataNode("res", INT32, INT32, F32, &rootResLevel_2);
	// // set dimenions from the new schedule information
	// weightInfo_2.setDims(32, 41); //
	// resInfo_2.setDims(-1, 41); // -1=N=232965, the number of nodes in the graph
 //    ffn_2.addInputData(&outputData_2);
 //    ffn_2.addInputData(&weightData_2);
 //    ffn_2.addOutputData(&resData_2);
 //    trainingLoop.addLoopNode(&ffn_2);
	// //* Dependencies
 //    auto inOutWeightDepRelationFeat_2 = RelationEdge(&outputData_2, ALL_RELATION, &resData_2, ALL_RELATION);
 //    auto inOutWeightDepRelationWeight_2 = RelationEdge(&weightData_2, COLS_RELATION, &resData_2, ROWS_RELATION);
 //    dependencies->push_back(&inOutWeightDepRelationFeat_2);
 //    dependencies->push_back(&inOutWeightDepRelationWeight_2);
 //    auto inOutWeightAssociation_2 = RelationEdge(&outputData_2, ROWS_RELATION, &weightData_2, COLS_RELATION);
 //    associations->push_back(&inOutWeightAssociation_2);
 //
	// // 2nd normalization calculation
	// auto normFeat2_2 = ForwardNode(UPDATE_NODE, ROW_BROADCAST_OP);
	// auto normFeat2Info_2 = DataInfo(RM_DTYPE);
	// normFeat2Info_2.setDims(-1, 32);
	// auto rootNormFeat2Level_2 = DataLevel(&normFeat2Info_2, true);
	// auto normFeat2Data_2 = DataNode("res", INT32, INT32, F32, &rootNormFeat2Level_2);
	// normFeat2_2.addInputData(&normData);
	// normFeat2_2.addInputData(&resData_2);
	// normFeat2_2.addOutputData(&normFeat2Data_2);
	// trainingLoop.addLoopNode(&normFeat2_2);
	// auto normFeat2NormDependency_2 = RelationEdge(&normData, ALL_RELATION, &normFeat2Data_2, ROWS_RELATION);
	// dependencies->push_back(&normFeat2NormDependency_2);
	// auto normFeat2FeatDependency_2 = RelationEdge(&resData_2, ALL_RELATION, &normFeat2Data_2, ALL_RELATION);
	// dependencies->push_back(&normFeat2FeatDependency_2);
	// auto normFeat2NormFeatAssociation_2 = RelationEdge(&normData, ALL_RELATION, &resData_2, ROWS_RELATION);
	// associations->push_back(&normFeat2NormFeatAssociation_2);
 //
	// // ReLU operation
	// auto logSoftmaxOp = ForwardNode(POINTWISE, NON_LNR_OP_LOG_SOFTMAX);
	// auto logSoftmaxInfo = DataInfo(RM_DTYPE);
	// logSoftmaxInfo.setDims(-1, 41);
	// auto rootLogSoftmaxLevel = DataLevel(&logSoftmaxInfo, true);
	// auto logSoftmaxData = DataNode("res", INT32, INT32, F32, &rootLogSoftmaxLevel);
	// logSoftmaxOp.addInputData(&normFeat2Data_2);
	// logSoftmaxOp.addOutputData(&logSoftmaxData);
	// trainingLoop.addLoopNode(&logSoftmaxOp);
	// auto logSoftmaxOpOnesDependency = RelationEdge(&normFeat2Data_2, ALL_RELATION, &logSoftmaxData, ALL_RELATION);
	// dependencies->push_back(&logSoftmaxOpOnesDependency);
 //
 //    // The entire program
 //    program->push_back(&loadDataset);
	// program->push_back(&trainingLoop);
 // //
	// double start, end;
	// start = get_time();
 //
	// auto ctx = new GALAContext(GPU_DEVICE, SINGLE_NODE_SINGLE);
	// std::string outputPath = "../test-codegen/";
	// auto genCode = CUDAGenerator(ctx, outputPath);
	// GALATransformations::complexityOperatorReordering(*program, *dependencies, *associations, *transforms);
	// GALATransformations::trainingInvariantCodeMotion(*program, *dependencies, *associations, *transforms);
	// GALATransformations::trainingSubGraph(*program, *dependencies, *associations, *transforms);
	// genCode.writeCode(*program, *dependencies, *associations, *transforms);
 //
	// end = get_time();
	// std::cout << "Time taken: " << (end - start)*1000  << std::endl;

    // Should be enough for now
	std::cout << "Works!" << std::endl;
}