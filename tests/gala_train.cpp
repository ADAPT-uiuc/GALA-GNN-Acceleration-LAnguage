//
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
#include "../src/codegen/common.h"

// Matrix classes
#include "../src/formats/dense_matrix.h"
#include "../src/formats/csrc_matrix.h"

// Frontend
#include "../src/frontend/context.h"

#include "../src/middle-end/middle-end.h"

extern void generate_ir();

extern FILE* yyin;
extern int yyparse();
ModelConfig m1;

std::vector<CIRNode*> GALAFEContext::program;
std::vector<RelationEdge*> GALAFEContext::dependencies;
std::vector<RelationEdge*> GALAFEContext::associations;
std::vector<TransformEdge*> GALAFEContext::transforms;

bool GALAFEContext::operator_reordering = true;
bool GALAFEContext::sparse_rewrites = true;
bool GALAFEContext::train_code_motion = true;
bool GALAFEContext::training_subgraph = true;
bool GALAFEContext::print_accuracy = false;

std::string GALAFEContext::opt_input = "";

//Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DMd_t;
//Dense matrix with integer values.
typedef DenseMatrix<ind1_t, ind2_t, val_int_t> DMi_t;
// Sparse matrix (graph)
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM_t;

/** RULES -- res input is always the 1st input for a computaiton op */
int main(int argc, char **argv) {
	std::string inputFile = argv[1];
	std::string outputPath = argv[2];

	m1 = ModelConfig();

	FILE *myfile = fopen(inputFile.c_str(), "r");
	if (!myfile) {
		std::cout << "Invalid File" << std::endl;
		return -1;
	}

	double start, end;
	start = get_time();

	yyin = myfile;
	yyparse();
	fclose(myfile);

	cout << " ---------------- printing model config ----------------------\n";
	cout << m1.to_string() << '\n';
	cout << "---------------------------------------------------------------\n";

	generate_ir();
	cout << " --------     checking generated ir output ------------ \n";
	cout << "PROGRAM (CIR Nodes): " << GALAFEContext::program.size() << '\n';

	for (int i = 0; i < GALAFEContext::program.size(); i++){
		cout << "        program node " << i << "\n";
	}
	auto p1 = dynamic_cast<ComputeNode*>(GALAFEContext::program[0]);
	std::cout << p1->getOutput(1)->getName() << " " << p1->getOutput(1)->getDataInfo()->getDirected() << std::endl;
	std::cout << p1->getOutput(1)->getName() << " " << p1->getOutput(1)->getDataInfo()->getWeighted() << std::endl;

	auto l1 = dynamic_cast<TrainingLoopNode*>(GALAFEContext::program[1]);
	auto o1 = l1->getNode(4);
	std::cout << "bb: " << o1->getOp() << " " << o1->getNumOpts() << std::endl;

	cout << "DEPENDENCIES " << GALAFEContext::dependencies.size() << '\n';
	for (int i = 0; i < GALAFEContext::dependencies.size(); i++){
		cout << "     dependency edge " << i << " with nodes " <<
			GALAFEContext::dependencies[i]->getNode1()->getName() <<
				", " << GALAFEContext::dependencies[i]->getNode2()->getName() << '\n';
	}
	std::cout << GALAFEContext::dependencies[1]->getNode1()->getName() << " " << GALAFEContext::dependencies[1]->getNode1()->getDataInfo()->getDirected() << std::endl;
	cout << "ASSOCIATIONS " << GALAFEContext::associations.size() << '\n';
	for (int i = 0; i < GALAFEContext::associations.size(); i++){
		cout << "     associations edge " << i << " with nodes " <<
			GALAFEContext::associations[i]->getNode1()->getName() <<
				", " << GALAFEContext::associations[i]->getNode2()->getName() << '\n';
	}
	std::cout << GALAFEContext::associations[0]->getNode1()->getName() << " " << GALAFEContext::associations[0]->getNode1()->getDataInfo()->getDirected() << std::endl;
	std::cout << GALAFEContext::associations[1]->getNode1()->getName() << " " << GALAFEContext::associations[1]->getNode1()->getDataInfo()->getDirected() << std::endl;
	cout << "TRANSFORMS " << GALAFEContext::transforms.size() << '\n';
	for (int i = 0; i < GALAFEContext::transforms.size(); i++){
		cout << "     transform edge " << i << " with nodes " <<
			GALAFEContext::transforms[i]->getNode1()->getName() <<
				", " << GALAFEContext::transforms[i]->getNode2()->getName() << '\n';
	}

	auto ctx = new GALAContext(GPU_DEVICE, SINGLE_NODE_SINGLE);
	auto genCode = CUDAGenerator(ctx, outputPath);
	if (GALAFEContext::operator_reordering)
	{
		GALATransformations::complexityOperatorReordering(GALAFEContext::program, GALAFEContext::dependencies,
		GALAFEContext::associations, GALAFEContext::transforms);
	}
	if (GALAFEContext::sparse_rewrites)
	{
		GALATransformations::sparsityAwareRewrites(GALAFEContext::program, GALAFEContext::dependencies,
		GALAFEContext::associations, GALAFEContext::transforms);
	}
	if (GALAFEContext::train_code_motion)
	{
		GALATransformations::trainingInvariantCodeMotion(GALAFEContext::program, GALAFEContext::dependencies,
			GALAFEContext::associations, GALAFEContext::transforms);
	}
	if (GALAFEContext::training_subgraph)
	{
		GALATransformations::trainingSubGraph(GALAFEContext::program, GALAFEContext::dependencies,
			GALAFEContext::associations, GALAFEContext::transforms);
	}
	genCode.writeCode(GALAFEContext::program, GALAFEContext::dependencies,
		GALAFEContext::associations, GALAFEContext::transforms);

	end = get_time();
	std::cout << "Time taken for GALA compilation: " << (end - start)*1000  << std::endl;
}