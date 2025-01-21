//
// Created by damitha on 4/23/24.
//
#ifndef GNN_ACCELERATION_LANGUAGE_COMMON_H
#define GNN_ACCELERATION_LANGUAGE_COMMON_H

#include <string>
#include <string>
#include <vector>
#include "../ir/compute.h"
#include <fstream>
#include <iostream>

// Specify the target device (Use to create the type of CodeGen instance)
enum Device
{
    CPU_DEVICE,
    GPU_DEVICE
};

// The distrubuted environment the GNN will operate on
enum Environment
{
    SINGLE_NODE_SINGLE, // Single node with a single CPU/GPU
    SINGLE_NODE_MULTI, // Single node with multiple CPUs/GPUs (Like the A100x4 Server)
    MULTI_NODE_SINGLE, // Distributed multi-node setting, each with a single CPU/GPU
    MULTI_NODE_MULTI // Distributed multi-node setting, each with multiple CPUs/GPUs
};

// Context of the execution used by GALA
// TODO: How to support heterogenous execution? Use multiple contexts for each function?
class GALAContext
{
private:
    Device device;
    Environment env;

public:
    GALAContext(Device dev, Environment env)
    {
        this->device = dev;
        this->env = env;
    }

    // Get the target device
    Device getDevice()
    {
        return this->device;
    }

    // Get the environment in terms of distribution
    Environment getEnv()
    {
        return this->env;
    }
};


// TODO -- For now both CmakeCode and KernelCode have the same operations
//  Separate this out. CMake is common for all, but the kernel code would also have
//  a PyTorch linking component
class Code
{
private:
    std::vector<std::string> codeLines;

public:
    Code()
    {
    };

    int getNum()
    {
        return this->codeLines.size();
    }

    void addCode(std::string& newCode)
    {
        this->codeLines.push_back(newCode);
    }

    std::string* atLine(int ix)
    {
        return &(this->codeLines.at(ix));
    }
};

class CMakeCode : public Code
{
};

class TargetCode : public Code
{
};

class KernelCode : public TargetCode
{
private:
    std::string name;
    std::vector<std::string> kernelCall;

public:
    KernelCode(std::string& name): TargetCode()
    {
        this->name = name;
    };

    std::string* getName()
    {
        return &this->name;
    }

    // Kernel call
    int getNumCall()
    {
        return this->kernelCall.size();
    }

    void addCallCode(std::string& newCode)
    {
        this->kernelCall.push_back(newCode);
    }

    std::string* atCallLine(int ix)
    {
        return &(this->kernelCall.at(ix));
    }

    void clearCall()
    {
        this->kernelCall.clear();
    }
};

// TODO break this down into multiple parts??
// Model code
class Model
{
private:
    // ID for the model
    std::string modelName;

    // These are in the main funciton
    // Model component definition (The init of a Torch model)
    Code modelUse;
    // Model Training Invariant
    Code modelInv;
    // Code for the model use
    Code modelTraining;
    Code modelValidation;
    Code modelTesting;

    // Code for the model defition
    // This is for the
    Code modelDef;
    // The code in the initialization section of the model + training invariant code
    Code modelInit;
    Code modelInitCall;
    // Model forward
    Code modelForward;
    Code modelForwardCall;


public:
    Model()
    {
        std::string defaultName = "gnn";
        this->modelName = defaultName;
    }

    Model(std::string& name)
    {
        this->modelName = name;
    }

    // TODO this is at the code generation phase so you don't need to clear / remove stuff
    //  Those should already have been decided in previous passes
    std::string* getName()
    {
        return &this->modelName;
    }

    // Def
    Code* getDef()
    {
        return &this->modelDef;
    }
    Code* getForwardCall()
    {
        return &this->modelForwardCall;
    }

    // Init
    Code* getInit()
    {
        return &this->modelInit;
    }
    Code* getInitCall()
    {
        return &this->modelInitCall;
    }

    // Use
    Code* getUse()
    {
        return &this->modelUse;
    }
    // Invariant
    Code* getInv()
    {
        return &this->modelInv;
    }

    // Forward
    Code* getForward()
    {
        return &this->modelForward;
    }

    // Training
    Code* getTraining()
    {
        return &this->modelTraining;
    }

    // Validation
    Code* getValidation()
    {
        return &this->modelValidation;
    }

    // Testing
    Code* getTesting()
    {
        return &this->modelTesting;
    }
};

// TODO You're generating code for Python. So you NEED to identify where to make any indents
class CodeGenerator
{
private:
    GALAContext* context;

protected:
    Code cmakeCode;
    Code importCode; // Imports
    Code kernelCode; // Kernels
    Code kernelCallCode; // Kernel calls
    Code autoGradCode; // Autograd code
    Code preCode; // Just one for now. Preprocessing should be done first and THEN the trasfers.
    Model model; // TODO: Assume a single model for now
    Code postCode; // Cleanup code?

    std::vector<std::string> generatedFunctions;

    std::ofstream outStreamModel;
    std::ofstream outStreamCMake;

public:
    CodeGenerator(GALAContext* context, std::string& outputPath)
    {
        this->context = context;
        this->openStream(outputPath);
    }

    void generateOpCode(ComputeNode* cNode, int& fcCount, bool outOfLoop = false)
    {
        if (cNode->getOp() == LOAD_OP)
        {
            // TODO assume a single load for now and make the index for it 0

            for (int oI = 0; oI < cNode->getNumOutputs(); oI++)
            {
                auto currentInfo = cNode->getOutput(oI)->getDataInfo();
                if (currentInfo->getFormat() == CSR_STYPE)
                {
                    currentInfo->setIndex(0);
                }
            }

            // This doesn't need to change
            std::string fileLoadCode = "    SM adj0;\n\
    std::string filename = \"" + cNode->getParam(0) +  "\";\n\
    readSM_npy32<SM>(filename, &adj0);\n\
\n\
    // Adj info\n\
    iT nrows = adj0.nrows();\n\
    global_nrows = nrows;\n\
    iT ncols = adj0.ncols();\n\
    nT nvals0 = adj0.nvals();\n\
\n\
    // Init input with random numbers\n\
    DM input_emb;\n\
    readDM_npy<DM>(filename + \"Feat.npy\", &input_emb,\n\
                   DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);\n\
    iT emb_size = input_emb.ncols();\n\
\n\
    DL labels;\n\
    readDM_npy<DL>(filename + \"Lab.npy\", &labels,\n\
                   DenseMatrix<ind1_t, ind2_t, lab_t>::DENSE_MTX_TYPE::RM);\n\
\n\
    DBL train_mask_load;\n\
    readDM_npy<DBL>(filename + \"TnMsk.npy\", &train_mask_load,\n\
                    DBL::DENSE_MTX_TYPE::RM);\n\
    DBL valid_mask_load;\n\
    readDM_npy<DBL>(filename + \"VlMsk.npy\", &valid_mask_load,\n\
                    DBL::DENSE_MTX_TYPE::RM);\n\
    DBL test_mask_load;\n\
    readDM_npy<DBL>(filename + \"TsMsk.npy\", &test_mask_load,\n\
                    DBL::DENSE_MTX_TYPE::RM);\n\
\n\
    DB train_mask;\n\
    repopulate<DBL, DB>(&train_mask_load, &train_mask);\n\
    DB valid_mask;\n\
    repopulate<DBL, DB>(&valid_mask_load, &valid_mask);\n\
    DB test_mask;\n\
    repopulate<DBL, DB>(&test_mask_load, &test_mask);\n\
    int classes =\n\
    *std::max_element(labels.vals_ptr(), labels.vals_ptr() + labels.nvals()) + 1;";

            preCode.addCode(fileLoadCode);
        } else if (cNode->getOpType() == AGGREGATE_NODE)
        {
            bool isColTile = false;

            // TODO get if col_tile from the data
             // Also add the autograd function which should be generated based on the program
             // TODO change the function call based on the optimizations
             //  Global_i@s_directed does not need to be passed. You do do neet to pass segments.
            std::string autoGradFunction = ""
"class GatherForward : public torch::autograd::Function<GatherForward> {\n\
public:\n\
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,\n\
                                 torch::Tensor input_dense, int li) {\n\
        ctx->saved_data[\"li\"] = li;\n\
        torch::Tensor offset_graph = global_offset_graph[2 * li];\n\
        torch::Tensor columns_graph = global_columns_graph[2 * li];\n\
        torch::Tensor value_graph = global_value_graph[2 * li];\n";

            if (isColTile){
                autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li];\n\
        int segments = global_segments[2 * li];\n\
         return gather_forward_gcn(input_dense, offset_graph, columns_graph,\n\
                            value_graph, bounds, global_nrows, segments);\n";
            } else
            {
                autoGradFunction += "        return gather_forward_gcn(input_dense, offset_graph, columns_graph,\n\
                                  value_graph);\n";
            }

            autoGradFunction += "    }\n\
\n\
    static torch::autograd::tensor_list\n\
    backward(torch::autograd::AutogradContext *ctx,\n\
             torch::autograd::tensor_list grad_outputs) {\n\
        torch::Tensor input_dense = grad_outputs[0];\n\
        int li = ctx->saved_data[\"li\"].toInt();\n\
        torch::Tensor offset_graph = global_offset_graph[2 * li + 1];\n\
        torch::Tensor columns_graph = global_columns_graph[2 * li + 1];\n\
        torch::Tensor value_graph = global_value_graph[2 * li + 1];\n";

            if (isColTile){
                autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li];\n\
        int segments = global_segments[2 * li];\n\
        return gather_forward_gcn(input_dense, offset_graph, columns_graph,\
                              value_graph, bounds, global_nrows, segments)";
            } else
            {
                autoGradFunction += "\
        return {gather_forward_gcn(input_dense, offset_graph, columns_graph,\n\
                                   value_graph), torch::Tensor()};\n";
            }

    autoGradFunction += "\
    }\n\
};";
        kernelCallCode.addCode(autoGradFunction);

        if (outOfLoop)
        {
            auto inGraphIndx = cNode->getInput(1)->getDataInfo()->getIndex();
            std::string tempForwardAggrCall = "t_iden = GatherForward::apply(t_iden, " + std::to_string(inGraphIndx) + ");";
            model.getInv()->addCode(tempForwardAggrCall);
        } else
        {
            auto inGraphIndx = cNode->getInput(1)->getDataInfo()->getIndex();
            std::string tempForwardAggrCall = "res = GatherForward::apply(res, " + std::to_string(inGraphIndx) + ");";
            model.getForward()->addCode(tempForwardAggrCall);
        }

        } else if (cNode->getOpType() == UPDATE_NODE)
        {
            if (fcCount == 0)
            {
                std::string inSize1 = "int size" + std::to_string(fcCount);
                model.getInitCall()->addCode(inSize1);

                std::string inSize2 = "int size" + std::to_string(fcCount + 1);
                model.getInitCall()->addCode(inSize2);

                std::string fcDef = "torch::nn::Linear fc" + std::to_string(fcCount) + "{nullptr};";
                model.getDef()->addCode(fcDef);

                std::string fcInit = "fc" + std::to_string(fcCount) + " = register_module(\"fc"
                + std::to_string(fcCount) + "\", torch::nn::Linear(size" + std::to_string(fcCount)
                + ", size" + std::to_string(fcCount + 1) + "));";
                model.getInit()->addCode(fcInit);

                // TODO need some way to add the inputs to the function call

                // TODO add the inputs to the forward call based on the actual inputs
                std::string tempForwardCall = "res = fc" + std::to_string(fcCount) + "->forward(res);";
                model.getForward()->addCode(tempForwardCall);
            } else
            {
                // TODO Fill this.
            }
            fcCount++;
        }
    }

// TODO Put this in the common codegen? Doesn't seem to have any context specific content yet
    void generateCode(std::vector<CIRNode*>& program)
    {
        // TODO add data transformations before data preparation.
        //  Should come from a middle end transformation.
        this->dataPrep(program);

        for (int i = 0; i < program.size(); i++)
        {
            int fcCount = 0;
            CIRNode* outNode = program[i];
            auto oNode = dynamic_cast<ComputeNode*>(outNode);
            if (oNode)
            {
                generateOpCode(oNode, fcCount, true);
            } else {
                std::string modelDef = "struct GALAGNN : torch::nn::Module {";
                model.getDef()->addCode(modelDef);
                std::string modelCall =  "GALAGNN(";
                model.getInitCall()->addCode(modelCall);

                // TODO generate this based on the program
                std::string tempFowradCall = "std::vector<torch::Tensor>\n\
forward(torch::Tensor input_dense,   // B\n\
      int nrows){";
                model.getForwardCall()->addCode(tempFowradCall);

                std::string resInit = "torch::Tensor res = input_dense;";
                model.getForward()->addCode(resInit);

                auto loopNode = dynamic_cast<TrainingLoopNode*>(outNode);

                std::string numIterCode = "int num_iters = " + std::to_string(loopNode->getIter()) + ";";
                preCode.addCode(numIterCode);

                for (int ix = 0; ix < loopNode->getLoopNodeNum(); ix++)
                {
                    CIRNode* inNode = loopNode->getNode(ix);
                    auto cNode = dynamic_cast<ComputeNode*>(inNode);
                    generateOpCode(cNode, fcCount);
                }
                std::string tempReturn = "return {torch::log_softmax(res, /*dim=*/1)};";
                model.getForward()->addCode(tempReturn);

                std::string closeForward = "    }\n"
                                           "};";
                model.getForward()->addCode(closeForward);

                std::string closeInit = "   }";
                model.getInit()->addCode(closeInit);

                std::string closeInitCall = "){\n";
                model.getInitCall()->addCode(closeInitCall);

                // TODO Change the embedding sizes based on the
                std::string tempModelInit = "auto net = std::make_shared<GALAGNN>(emb_size, classes);";
                model.getUse()->addCode(tempModelInit);

                std::string modelTransfer = "net->to(device);";
                model.getUse()->addCode(modelTransfer);

                if (loopNode->getOptimizer() == ADAM)
                {
                    std::string optmCode = "torch::optim::Adam optimizer(\n\
net->parameters(), torch::optim::AdamOptions(1e-2).weight_decay(5e-4));";
                    model.getUse()->addCode(optmCode);
                } else
                {
                    std::cout << "Optimizer not supported." << std::endl;
                }

                // TODO generate this using the test loop
                std::string tempTrainLoop = "for (size_t epoch = 1; epoch <= num_iters; ++epoch) {\n\
    // Reset gradients.\n\
    optimizer.zero_grad();\n\
    torch::Tensor prediction =\n\
        net->forward(t_iden, nrows)[0];\n\
    torch::Tensor prediction_train = prediction.index({t_train_mask});\n\
    torch::Tensor labels_train = t_labs.index({t_train_mask});\n\
    auto criterion = torch::nn::CrossEntropyLoss();\n\
    torch::Tensor d_loss = criterion(prediction_train, labels_train);\n\
    d_loss.backward();\n\
    optimizer.step();\n\
    torch::Tensor prediction_test = prediction.index({t_test_mask});\n\
    torch::Tensor labels_test = t_labs.index({t_test_mask});\n\
    auto [pred_val, pred_idx] = torch::max({prediction_test}, 1);\n\
    auto correct = torch::sum(pred_idx == labels_test);\n\
    std::cout << \"Epoch \" << epoch << \" Loss: \" << d_loss.item<val_t>()\n\
              << \" Accuracy: \"\n\
              << (correct.item<val_t>() * 100.0 / labels_test.sizes()[0])\n\
              << std::endl;\n\
}";
                model.getUse()->addCode(tempTrainLoop);
            }
        }

        std::string closeMain = "}";
        postCode.addCode(closeMain);
    }

    GALAContext* getContext()
    {
        return this->context;
    }

    // Handle the stream to write to
    void openStream(std::string& outputPath)
    {
        std::string cmakePath = outputPath + "CMakeLists.txt";
        this->outStreamCMake = std::ofstream(cmakePath);

        std::string modelPath = outputPath + "gala.cu";
        this->outStreamModel = std::ofstream(modelPath);
    }

    void closeStream()
    {
        this->outStreamModel.close();
        this->outStreamCMake.close();
    }

    void writeCode(Code &code, std::ofstream &outStream, const std::string &end = "\n", bool skipFirstEnd = false,
        bool skip2ndLastEnd = false){
        for (int ix = 0; ix < code.getNum(); ix++){
            auto codeLine = code.atLine(ix);
            outStream << *codeLine;
            // skip adding end to first
            if (skipFirstEnd and ix == 0)
            {
                continue;
            } else if (skip2ndLastEnd and ix >= code.getNum() - 2)
            {
                continue;
            } else
            {
                outStream << end;
            }
        }
    }

    //    void addImport(PyCodeLine* pyCode){
    //        this->moreImports.push_back(pyCode);
    //    }
    //    void addPreCode(PyCodeLine* pyCode){
    //        this->preCode.push_back(pyCode);
    //    }
    //    void addPostCode(PyCodeLine* pyCode){
    //        this->postCode.push_back(pyCode);
    //    }

    // Separate function so it can be extended in the architecture specific components
    virtual void initCMake()
    {
    };

    // Separate function so it can be extended in the architecture specific components
    virtual void initKernels(std::vector<CIRNode*>& program)
    {
    };

    virtual void dataPrep(std::vector<CIRNode*>& program)
    {
    };

    void commonPerCode(std::vector<CIRNode*>& program)
    {
        // Will not change for now
        // TODO need to change the types based on the data
        std::string tempStdCommon = "#include <algorithm>\n\
typedef int ind1_t;\n\
typedef int ind2_t;\n\
typedef long lab_t;\n\
typedef float val_t;\n\
typedef int mask_load_t;\n\
typedef bool mask_t;\n\
// Dense matrix with double values.\n\
typedef DenseMatrix<ind1_t, ind2_t, val_t> DM;\n\
typedef DenseMatrix<ind1_t, ind2_t, lab_t> DL;\n\
typedef DenseMatrix<ind1_t, ind2_t, mask_load_t> DBL;\n\
typedef DenseMatrix<ind1_t, ind2_t, mask_t> DB;\n\
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM;\n\
int global_nrows;\n\
std::vector<int> global_segments;\n\
bool global_is_directed;\n\
\n\
std::vector<torch::Tensor> global_offset_graph;\n\
std::vector<torch::Tensor> global_columns_graph;\n\
std::vector<torch::Tensor> global_value_graph;\n\
std::vector<torch::Tensor> global_bounds;";
        importCode.addCode(tempStdCommon);

        std::string mainFuncCode  = "int main(int argc, char **argv) {\n\
  typedef typename SM::itype iT;\n\
  typedef typename SM::ntype nT;\n\
  typedef typename SM::vtype vT;\n\
\n\
  typedef typename DM::itype diT;\n\
  typedef typename DM::ntype dnT;\n\
  typedef typename DM::vtype dvT;";
        preCode.addCode(mainFuncCode);
    }

    void writeCode(std::vector<CIRNode*> &program)
    {
        // Kernel code - Architecture dependant
        // CMake (also has write for now?)
        initCMake();
        // Kernels
        initKernels(program);
        commonPerCode(program);

        generateCode(program);

        this->writeCode(cmakeCode, outStreamCMake);
        this->writeCode(importCode, outStreamModel);
        this->writeCode(kernelCode, outStreamModel);
        this->writeCode(kernelCallCode, outStreamModel);
        this->writeCode(*model.getDef(), outStreamModel);
        this->writeCode(*model.getInitCall(), outStreamModel, ", ", true, true);
        this->writeCode(*model.getInit(), outStreamModel);
        this->writeCode(*model.getForwardCall(), outStreamModel);
        this->writeCode(*model.getForward(), outStreamModel);
        this->writeCode(preCode, outStreamModel);
        this->writeCode(*model.getInv(), outStreamModel);
        this->writeCode(*model.getUse(), outStreamModel);
        this->writeCode(postCode, outStreamModel);

        this->closeStream();
    }
};

#endif //GNN_ACCELERATION_LANGUAGE_COMMON_H
