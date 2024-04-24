//
// Created by damitha on 4/23/24.
//

#ifndef GNN_ACCELERATION_LANGUAGE_CUDA_H
#define GNN_ACCELERATION_LANGUAGE_CUDA_H

#include <fstream>
#include "common.h"

class GenerateCUDA : public CodeGenerator {
private:
public:
    GenerateCUDA(GALAContext *context, std::string *outputPath) : CodeGenerator(context, outputPath) {}

    void generateCode(std::vector<ComputeNode *> program) {
        auto conxt = this->getContext();

        if (conxt->getEnv() == SINGLE_NODE_SINGLE) {
            for (int pt = 0; pt < program.size(); pt++) {
                auto current = program.at(pt);
                if (current->getOp() == TRAIN_CONTROL) {
                    // TODO Introduce a loop for training code generation

                    auto trainNode = static_cast<TrainingLoopNode *>(current);
                    for (int lpt = 0; lpt < trainNode->getLoopNodeNum(); lpt++) {
                        // TODO Handle each individual op
                    }

                    // TODO Close the loop
                } else if (current->getOp() == IF_CONTROL) {
                    // TODO Need to add
                    std::cout << "Skip for now in code generation" << std::endl;
                } else {
                    // TODO handle each individual op
                }
            }
        } else {
            std::cout << "Only single node single device is supported for now" << std::endl;
        }

    }

    void generateStatement(ComputeNode &node) {

    }

    void generateLoad(ComputeNode &node) {

    }
};

#endif //GNN_ACCELERATION_LANGUAGE_CUDA_H
