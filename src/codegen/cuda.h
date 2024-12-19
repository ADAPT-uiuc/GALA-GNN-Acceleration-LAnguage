//
// Created by damitha on 4/23/24.
//

#ifndef GNN_ACCELERATION_LANGUAGE_CUDA_H
#define GNN_ACCELERATION_LANGUAGE_CUDA_H

#include "common.h"

class CUDAGenerator : public CodeGenerator
{
public:
    CUDAGenerator(GALAContext* context, std::string& outputPath) : CodeGenerator(context, outputPath)
    {
    }

    void initCMake() override
    {
        std::string cmakeCudaBase = "cmake_minimum_required(VERSION 3.1 FATAL_ERROR)\n"
            "project(gala_cuda LANGUAGES CUDA CXX)\n"
            "set(CMAKE_CXX_COMPILER icpx)\n"
            "find_package(Torch REQUIRED)\n"
            "find_package(OpenMP)\n"
            "include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})\n"
            "link_libraries(\"${TORCH_LIBRARIES}\" cudart cusparse)\n"
            "add_compile_options(-Xcompiler -fopenmp -march=native -O3)\n"
            "add_compile_definitions(GALA_TORCH)\n"
            "add_compile_definitions(GN_1)\n"
            "add_compile_definitions(PT_0)\n"
            "add_compile_definitions(ST_0)\n"
            "add_compile_definitions(A_ALLOC)";
        std::string cmakeExecutable = "add_executable(gala_model gala.cu)\n"
            "target_compile_features(gala PRIVATE cxx_std_14)";
        cmakeCode.addCode(cmakeCudaBase);
        cmakeCode.addCode(cmakeExecutable);
    }

    void initKernels() override
    {
        std::string cudaInitFunctions = "#define CUDA_CHECK(func)                      \
          do {                                                                         \
            cudaError_t status = (func);                                               \
            if (status != cudaSuccess) {                                               \
              printf(\"CUDA API failed at line %d with error: %s (%d)\n\", __LINE__,   \
                     cudaGetErrorString(status), status);                              \
              exit(EXIT_FAILURE);                                                      \
            }                                                                          \
          } while (0) \
        \
        #define CUSPARSE_CHECK(func)                                                   \
          do {                                                                         \
            cusparseStatus_t status = (func);                                          \
            if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
              printf(\"CUSPARSE failed at line %d with error: %s (%d)\n\", __LINE__,   \
                     cusparseGetErrorString(status), status);                          \
              exit(EXIT_FAILURE);                                                      \
            }                                                                          \
          } while (0)";
        preCode.addCode(cudaInitFunctions);
    }

    // TODO Put this in the common codegen? Doesn't seem to have any context specific content yet
    void generateCode(std::vector<ComputeNode*>& program) override
    {
        //        int iIndent = 0;
        //        auto conxt = this->getContext();
        //
        //        if (conxt->getEnv() == SINGLE_NODE_SINGLE) {
        //            for (int pt = 0; pt < program.size(); pt++) {
        //                auto current = program.at(pt);
        //                if (current->getOp() == TRAIN_CONTROL) {
        //                    // TODO Introduce a loop for training code generation
        //
        //                    auto trainNode = static_cast<TrainingLoopNode *>(current);
        //                    for (int lpt = 0; lpt < trainNode->getLoopNodeNum(); lpt++) {
        //                        // TODO Handle each individual op
        //                        auto stmntNode = static_cast<StatementNode *>(trainNode->getNode(lpt));
        //                        generateStatement(*stmntNode);
        //                    }
        //
        //                    // TODO Close the loop
        //                } else if (current->getOp() == IF_CONTROL) {
        //                    // TODO Need to add
        //                    std::cout << "Skip for now in code generation" << std::endl;
        //                } else {
        //                    // TODO handle each individual op
        //                    auto stmntNode = static_cast<StatementNode *>(current);
        //                    generateStatement(iIndent, *stmntNode);
        //                }
        //            }
        //        } else {
        //            std::cout << "Only single node single device is supported for now" << std::endl;
        //        }
        //
        //        // Write the code to the output file/s
        //        this->writeCode();
    }

    //    void generateStatement(int inden, StatementNode &node) {
    //        if (node.getOp() == LOAD_OP){
    //            generateLoad(inden, node);
    //        } else {
    //            std::cout << "To be supported in the future" << std::endl;
    //        }
    //    }

    //    // Move to Common
    //    void generateLoad(int inden, StatementNode &node) {
    //        if (node.getParam(0) == "Reddit"){
    //            std::string sl1("from dgl.data import RedditDataset");
    //            auto pl1 = PyCodeLine(inden, sl1);
    //            this->addImport(&pl1);
    //            std::string sl2("dataset_name = getattr(dgl.data, \"RedditDataset\", False)");
    //            auto pl2 = PyCodeLine(inden, sl2);
    //            this->addImport(&pl2);
    //            std::string sl3("dataset = dataset_name()");
    //            auto pl3 = PyCodeLine(inden, sl3);
    //            this->addImport(&pl3);
    //            std::string sl4("graph = dataset[0]");
    //            auto pl4 = PyCodeLine(inden, sl4);
    //            this->addImport(&pl4);
    //        } else {
    //            std::cout << "To be supported in the future" << std::endl;
    //        }
    //        // TODO Map the loaded data into variables
    //        // TODO Add labels + mask to load as well in diagram / IR
    //        // Load common components -- Getting the features etc
    //        addCode("input_dense = graph.ndata[\"feat\"]");
    //        addCode("labels = graph.ndata[\"label\"]");
    //
    //        // TODO Remove from here. This is for GCN norm
    //        addImport("from dgl.utils import expand_as_pair");
    //        addCode("feat_src, feat_dst = expand_as_pair(input_dense, graph)");
    //        addCode("degs = graph.out_degrees().to(feat_src).clamp(min=1)");
    //        addCode("norm = torch.pow(degs, -0.5)");
    //        addCode("graph.srcdata.update({\"di\": norm})");
    //        addCode("graph.dstdata.update({\"do\": norm})");
    //        addCode("graph.apply_edges(fn.u_mul_v('di', 'do', 'dd'))");
    //
    //        // TODO Need a way to track this.
    //        //  A data object to a name. - Graph is x, cols n offsets are y in the original data matrix
    //        addCode("offsets = graph.adj_tensors('csr')[0].to(torch.int32)");
    //        addCode("cols = graph.adj_tensors('csr')[1].to(torch.int32)");
    //        // TODO Remove from here. This is for GCN norm
    //        addCode("vals = graph.edata['dd']");
    //
    //        addCode("");
    //    }
};

#endif //GNN_ACCELERATION_LANGUAGE_CUDA_H
