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

// Matrix classes
#include "../src/formats/dense_matrix.h"
#include "../src/formats/csrc_matrix.h"

// Frontend
#include "../src/frontend/context.h"

extern void generate_ir();

extern FILE* yyin;
extern int yyparse();
ModelConfig m1;

/** RULES -- res input is always the 1st input for a computaiton op */
int main(int argc, char **argv) {
	const char* graph = argv[1];
	const char* model = argv[2];
	const char* hw = argv[3];

	std::string inputGALAFrontEnd = "../tests/GALA-DSL/" + model + "/" + graph + "/" + hw + ".txt";

	m1 = ModelConfig();

	FILE *myfile = fopen(inputGALAFrontEnd.c_str(), "r");
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
	cout << "DEPENDENCIES " << GALAFEContext::dependencies.size() << '\n';
	for (int i = 0; i < GALAFEContext::dependencies.size(); i++){
		cout << "     dependency edge " << i << " with nodes " <<
			GALAFEContext::dependencies[i]->getNode1()->getName() <<
				", " << GALAFEContext::dependencies[i]->getNode2()->getName() << '\n';
	}
	cout << "ASSOCIATIONS " << GALAFEContext::associations.size() << '\n';
	for (int i = 0; i < GALAFEContext::associations.size(); i++){
		cout << "     associations edge " << i << " with nodes " <<
			GALAFEContext::associations[i]->getNode1()->getName() <<
				", " << GALAFEContext::associations[i]->getNode2()->getName() << '\n';
	}
	cout << "TRANSFORMS " << GALAFEContext::transforms.size() << '\n';
	for (int i = 0; i < GALAFEContext::transforms.size(); i++){
		cout << "     transform edge " << i << " with nodes " <<
			GALAFEContext::transforms[i]->getNode1()->getName() <<
				", " << GALAFEContext::transforms[i]->getNode2()->getName() << '\n';
	}

	auto ctx = new GALAContext(GPU_DEVICE, SINGLE_NODE_SINGLE);
	std::string outputPath = "../test-codegen/";
	auto genCode = CUDAGenerator(ctx, outputPath);
	GALATransformations::complexityOperatorReordering(GALAFEContext::program, GALAFEContext::dependencies,
		GALAFEContext::associations, GALAFEContext::transforms);
	GALATransformations::trainingInvariantCodeMotion(GALAFEContext::program, GALAFEContext::dependencies,
		GALAFEContext::associations, GALAFEContext::transforms);
	GALATransformations::trainingSubGraph(GALAFEContext::program, GALAFEContext::dependencies,
		GALAFEContext::associations, GALAFEContext::transforms);
	genCode.writeCode(GALAFEContext::program, GALAFEContext::dependencies,
		GALAFEContext::associations, GALAFEContext::transforms);

	end = get_time();
	std::cout << "Time taken for GALA compilation: " << (end - start)*1000  << std::endl;
}