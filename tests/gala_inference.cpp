//
#include "gala_driver.h"

std::vector<CIRNode*> GALAFEContext::program;
std::vector<RelationEdge*> GALAFEContext::dependencies;
std::vector<RelationEdge*> GALAFEContext::associations;
std::vector<TransformEdge*> GALAFEContext::transforms;

bool GALAFEContext::operator_reordering = true;
bool GALAFEContext::sparse_rewrites = true;
bool GALAFEContext::train_code_motion = true;
bool GALAFEContext::training_subgraph = true;
bool GALAFEContext::print_accuracy = false;
bool GALAFEContext::print_memory = false;
bool GALAFEContext::use_long = false;

std::string GALAFEContext::opt_input = "";

ModelConfig m1;

int main(int argc, char **argv) {
	return gala_run(argc, argv, false, 5);
}
