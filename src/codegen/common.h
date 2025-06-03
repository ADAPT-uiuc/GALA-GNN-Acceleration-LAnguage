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
    Code modelPreCall;
    Code modelCall;
    Code modelPostCall;
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
    Code modelForwardCallPre;
    Code modelForwardCallInternal;
    Code modelForwardCallPost;


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
    Code* getForwardCallPre()
    {
        return &this->modelForwardCallPre;
    }
    Code* getForwardCallInternal()
    {
        return &this->modelForwardCallInternal;
    }
    Code* getForwardCallPost()
    {
        return &this->modelForwardCallPost;
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
    Code* getPreCall()
    {
        return &this->modelPreCall;
    }
    Code* getCall()
    {
        return &this->modelCall;
    }
    Code* getPostCall()
    {
        return &this->modelPostCall;
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


    std::string processDims(int val)
    {
        if (val < 0)
        {
            if (val == -1)
            {
                return "global_nrows";
            } else if (val == -2)
            {
                return "global_classes";
            } else if (val == -3)
            {
                return "global_emb_size";
            } else
            {
                // TODO
                return "ERROR!!!";
            }
        } else
        {
            return std::to_string(val);
        }
    }

    // Move to data node 
    bool hasDOpt(DataNode* dNode, DataOptimization op)
    {
        auto opts = dNode->getDataInfo()->getOpts();
        for (int ix = 0; ix < opts->size(); ix++)
        {
            auto opt = opts->at(ix);
            if (opt.first == op)
            {
                return true;
            }
        }
        return false;
    }

    // TODO Move to compute node
    bool hasCOpt(ComputeNode* cNode, CompOptimization op)
    {
        for (int ix = 0; ix < cNode->getNumOpts(); ix++)
        {
            auto opt = cNode->getOpt(ix);
            if (opt->first == op)
            {
                return true;
            }
        }
        return false;
    }

    std::string getKernelName(ComputeNode* cNode)
    {
        std::string kernelName = "";
        if (cNode->getOp() == AGGREGATE_MUL_SUM_OP)
        {
            kernelName += "aggregate_node_mul_sum";
        } else if (cNode->getOp() == AGGREGATE_MUL_SUM_DIRECT)
        {
            kernelName += "aggregate_node_mul_sum_direct";
        } else if (cNode->getOp() == NON_LNR_OP_SOFTMAX)
        {
            kernelName += "non_lnr_op_softmax";
        } else if (cNode->getOp() == AGGREGATE_EDGE_MUL_OP)
        {
            kernelName += "aggregate_edge_mul";
        } else if (cNode->getOp() == AGGREGATE_EDGE_SUM_OP)
        {
            kernelName += "aggregate_edge_sum";
        } else if (cNode->getOp() == LOAD_OP)
        {
            kernelName += "load_op";
        } else if (cNode->getOp() == ONES_OP)
        {
            kernelName += "ones_op";
        } else
        {
            kernelName += "unsupported";
        }
        // TODO add other kernel optimizations
        for (std::pair<CompOptimization, float> optPair: *cNode->getOpts())
        {
            if (optPair.first == COARSE_COPT)
            {
                kernelName += "_coarse" + std::to_string((int)(optPair.second));
            }
        }
        return kernelName;
    }

    std::string generateOutputString(ComputeNode* cNode, bool outOfLoop)
    {
        for (int ix = 0; ix < cNode->getNumInputs(); ix++)
        {
            if (cNode->getOutput(0)->getName() == cNode->getInput(ix)->getName())
            {
                return cNode->getInput(ix)->getName();
            }
        }
         if (outOfLoop)
         {
             return "torch::Tensor " + cNode->getOutput(0)->getName();
         } else
         {
             return cNode->getOutput(0)->getName();
         }

    }

    std::string generateTransformation(DataNode* srcNode, std::vector<TransformEdge*>& transforms)
    {
        std::string resString = "";
        for (int ix = 0; ix < transforms.size(); ix++)
        {
            auto transform = transforms[ix];
            if (transform->getNode1()->getName() == srcNode->getName())
            {
                auto dNode = transform->getNode2();
                auto tr = transform->getTransformation(0);
                if (tr->getTransformation() == COL_TILE_DOPT)
                {
                    resString +=  "  std::vector<SM *> tiled_" + dNode->getName() +";\n\
  tiled_" + dNode->getName() + ".push_back(&" + srcNode->getName() + ");\n\
  torch::Tensor total_offsets_" + dNode->getName() + ";\n\
  torch::Tensor total_cols_" + dNode->getName() + ";\n\
  torch::Tensor total_vals_" + dNode->getName() + ";\n\
  torch::Tensor total_bounds_" + dNode->getName() + ";\n\
  std::vector<iT> tile_offsets_" + dNode->getName() + " =\n\
    static_ord_col_breakpoints<SM>(&" + srcNode->getName() + ", " + tr->getParam(0) +");\n\
  iT segments_" + dNode->getName() + " = tile_offsets_" + dNode->getName() + ".size() - 1;\n\
  total_offsets_" + dNode->getName() + " = torch::zeros({(" + srcNode->getName() + ".nrows() + 1) * (segments_"
                    + dNode->getName() + ")}, options_int_tile);\n\
  total_cols_" + dNode->getName() + " = torch::zeros({" + srcNode->getName() + ".nvals()}, options_int_tile);\n\
  total_vals_" + dNode->getName() + " = torch::zeros({" + srcNode->getName() + ".nvals()}, options_float_tile);\n\
  total_bounds_" + dNode->getName() + " = torch::zeros({2 * (segments_" + dNode->getName() + ")}, options_int_tile);\n\
  ord_col_tiling_torch(tile_offsets_" + dNode->getName() + ", total_offsets_" + dNode->getName() +
                        ", total_cols_" + dNode->getName() + ", total_vals_" + dNode->getName() + ",\n\
    total_bounds_" + dNode->getName() + ", &" + srcNode->getName() + ");\n\
  iT *offset_ptr_" + dNode->getName() + " = total_offsets_" + dNode->getName() + ".data_ptr<iT>();\n\
  iT *col_ptr_" + dNode->getName() + " = total_cols_" + dNode->getName() + ".data_ptr<iT>();\n\
  vT *val_ptr_" + dNode->getName() + " = total_vals_" + dNode->getName() + ".data_ptr<vT>();\n";

                    resString += "  global_segments.push_back(segments_" + dNode->getName() + ");\n";
                    resString += "  global_bounds.push_back(total_bounds_" + dNode->getName() + ");\n";
                    if (!dNode->getDataInfo()->getDirected())
                    {
                        resString += "  global_segments.push_back(segments_" + dNode->getName() + ");\n";
                        resString += "  global_bounds.push_back(total_bounds_" + dNode->getName() + ");\n";
                    } else
                    {
                        std::string tilingParam;
                        if (tr->getNumParam() > 1)
                        {
                            tilingParam = tr->getParam(1);
                        } else
                        {
                            tilingParam = tr->getParam(0);
                        }
                        resString +=  "  std::vector<SM *> tiled_" + dNode->getName() +"_b;\n\
  tiled_" + dNode->getName() + "_b.push_back(&" + srcNode->getName() + "_b);\n\
  torch::Tensor total_offsets_" + dNode->getName() + "_b;\n\
  torch::Tensor total_cols_" + dNode->getName() + "_b;\n\
  torch::Tensor total_vals_" + dNode->getName() + "_b;\n\
  torch::Tensor total_bounds_" + dNode->getName() + "_b;\n\
  std::vector<iT> tile_offsets_" + dNode->getName() + "_b =\n\
    static_ord_col_breakpoints<SM>(&" + srcNode->getName() + "_b, " + tilingParam +");\n\
  iT segments_" + dNode->getName() + "_b = tile_offsets_" + dNode->getName() + "_b.size() - 1;\n\
  total_offsets_" + dNode->getName() + "_b = torch::zeros({(" + srcNode->getName() + "_b.nrows() + 1) * (segments_"
                    + dNode->getName() + "_b)}, options_int_tile);\n\
  total_cols_" + dNode->getName() + "_b = torch::zeros({" + srcNode->getName() + "_b.nvals()}, options_int_tile);\n\
  total_vals_" + dNode->getName() + "_b = torch::zeros({" + srcNode->getName() + "_b.nvals()}, options_float_tile);\n\
  total_bounds_" + dNode->getName() + "_b = torch::zeros({2 * (segments_" + dNode->getName() + "_b)}, options_int_tile);\n\
  ord_col_tiling_torch(tile_offsets_" + dNode->getName() + "_b, total_offsets_" + dNode->getName() +
                        "_b, total_cols_" + dNode->getName() + "_b, total_vals_" + dNode->getName() + "_b,\n\
    total_bounds_" + dNode->getName() + "_b, &" + srcNode->getName() + "_b);\n\
  iT *offset_ptr_" + dNode->getName() + "_b = total_offsets_" + dNode->getName() + "_b.data_ptr<iT>();\n\
  iT *col_ptr_" + dNode->getName() + "_b = total_cols_" + dNode->getName() + "_b.data_ptr<iT>();\n\
  vT *val_ptr_" + dNode->getName() + "_b = total_vals_" + dNode->getName() + "_b.data_ptr<vT>();\n";

                        resString += "  global_segments.push_back(segments_" + dNode->getName() + "_b);\n";
                        resString += "  global_bounds.push_back(total_bounds_" + dNode->getName() + "_b);\n";
                    }
                } else if (tr->getTransformation() == SUBGRAPH_DOPT)
                {
                    if (tr->getNumParam() == 2)
                    {
                        resString +=  " std::vector<SM *> forward_adj;\n\
  std::vector<SM *> backward_adj;\n\
  getMaskSubgraphs(&adj0, &train_mask, " + tr->getParam(1) + ", forward_adj, backward_adj);\n";
                        for (int i = 0; i < std::stoi(tr->getParam(1)); i++)
                        {
                            int iy = std::stoi(tr->getParam(1)) - (i + 1);
                            int iz = i + 1;
                            resString += "  SM adj" + std::to_string(iz) + " = *forward_adj[" + std::to_string(iy) +"];\n\
  SM adj" + std::to_string(iz) + "_b = *backward_adj[" + std::to_string(iy) +"];\n\
  nT nvals" + std::to_string(iz) + " = adj" + std::to_string(iz) + ".nvals();\n";
                        }
                    }
                    resString += generateTransformation(dNode, transforms);
                } else if (tr->getTransformation() == SAMPLE_DOPT)
                {
                    resString += "inplace_sample_graph(&" + srcNode->getName() + ", " +  tr->getParam(0) + ");";
                    resString += generateTransformation(dNode, transforms);
                }
            }
        }
        return resString;

    }

    void generateOpCode(ComputeNode* cNode, int& fcCount, int& fcEdgeCount, int& epCount, bool outOfLoop, bool& hasFFNEdgeUpdate,
        std::unordered_set<std::string> &encounteredAutograds,
        std::vector<int> &inputSizes,
        std::vector<TransformEdge*>& transforms)
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
    *std::max_element(labels.vals_ptr(), labels.vals_ptr() + labels.nvals()) + 1;\n\
    global_classes = classes;\n\
    global_emb_size = emb_size;";
            preCode.addCode(fileLoadCode);

            // Graph output
            auto outputGraph = cNode->getOutput(1);
            std::string transformationCode = generateTransformation(outputGraph, transforms);
            preCode.addCode(transformationCode);
        } else if (cNode->getOp() == AGGREGATE_EDGE_SUM_OP)
        {
            hasFFNEdgeUpdate = true;
            bool isColTile = hasDOpt(cNode->getInput(2), COL_TILE_DOPT);

            if (encounteredAutograds.find(getKernelName(cNode)) == encounteredAutograds.end())
            {
                encounteredAutograds.insert(getKernelName(cNode));
                std::string autoGradFunction = "class " + getKernelName(cNode) + "_AutoGrad : public torch::autograd::Function<" + getKernelName(cNode) + "_AutoGrad> {\n\
public:\n\
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,\n\
                               torch::Tensor input_dense1,\n\
                               torch::Tensor input_dense2,\n\
                               int li) {\n";
                autoGradFunction += "        ctx->saved_data[\"li\"] = li;\n\
        torch::Tensor offset_graph = global_offset_graph[2 * li];\n\
        torch::Tensor columns_graph = global_columns_graph[2 * li];\n\
        torch::Tensor value_graph = global_value_graph[2 * li];\n";

                if (isColTile){
                    autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li];\n\
        int segments = global_segments[2 * li];\n\
        return edge_sddvv(input_dense1, input_dense2, offset_graph, columns_graph,\n\
                            value_graph, bounds, global_nrows, segments);";
                } else
                {
                    autoGradFunction += "unsupported\n";
                }
                autoGradFunction += "}\n";

                autoGradFunction += " static torch::autograd::tensor_list\n\
                    backward(torch::autograd::AutogradContext *ctx,\n\
                             torch::autograd::tensor_list grad_outputs) {\n\
                    torch::Tensor d_value_graph = grad_outputs[0];\n";
                autoGradFunction += " int li = ctx->saved_data[\"li\"].toInt();\n\
        torch::Tensor offset_graph = global_offset_graph[2 * li + 1];\n\
        torch::Tensor columns_graph = global_columns_graph[2 * li + 1];\n\
        torch::Tensor bounds = global_bounds[2 * li + 1];\n\
        int segments = global_segments[2 * li + 1];\n\
        torch::Tensor back_res = node_spmv_backward_of_sddmm_eaggr(\n\
                    offset_graph, columns_graph, // This should be the reverse graph\n\
                    d_value_graph, bounds, global_nrows, segments);\n\
        return {back_res,\n\
                back_res,\n\
                torch::Tensor()};\n\
    }\n\
};\n";
                kernelCallCode.addCode(autoGradFunction);
            }
            // TODO change the names from the input
            auto inGraphIndx = cNode->getInput(2)->getDataInfo()->getIndex();
            std::string tempForwardAggrCall = generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
            + "_AutoGrad::apply(" + cNode->getInput(0)->getName() + ", " + cNode->getInput(1)->getName() + ", " + std::to_string(inGraphIndx) + ");";
            model.getForward()->addCode(tempForwardAggrCall);
        } else if (cNode->getOp() == NON_LNR_OP_SOFTMAX)
        {
            bool isColTile = hasDOpt(cNode->getInput(0), COL_TILE_DOPT);

            if (encounteredAutograds.find(getKernelName(cNode)) == encounteredAutograds.end())
            {
                encounteredAutograds.insert(getKernelName(cNode));
                std::string autoGradFunction = "class " + getKernelName(cNode) + "_AutoGrad : public torch::autograd::Function<" + getKernelName(cNode) + "_AutoGrad> {\n\
public:\n\
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,\n\
                               torch::Tensor value_graph,\n\
                               int li) {\n";
                autoGradFunction += "        ctx->saved_data[\"li\"] = li;\n\
        torch::Tensor offset_graph = global_offset_graph[2 * li];\n\
        torch::Tensor columns_graph = global_columns_graph[2 * li];\n";

                if (isColTile){
                    autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li];\n\
        int segments = global_segments[2 * li];\n";
                } else
                {
                    autoGradFunction += "unsupported\n";
                }

                autoGradFunction += "    torch::Tensor val_exp = torch::exp(value_graph);\n\
    val_exp = torch::clamp(val_exp, 0.0, 1e12);\n\
    torch::Tensor row_sum = node_spmv_backward_of_sddmm_nln(\n\
        offset_graph, columns_graph, val_exp, bounds, global_nrows,\n\
        segments);\n\
    auto options = torch::TensorOptions()\n\
                       .dtype(torch::kFloat)\n\
                       .requires_grad(true)\n\
                       .device(torch::kCUDA, 0);\n\
    row_sum = torch::reciprocal(row_sum);\n\
    val_exp = inplace_softmax_sddvv(row_sum, offset_graph, columns_graph, \n\
                                    val_exp, bounds, global_nrows, segments);\n\
    ctx->save_for_backward({val_exp});\n\
    return val_exp;\n\
  }\n\
  static torch::autograd::tensor_list\n\
  backward(torch::autograd::AutogradContext *ctx,\n\
           torch::autograd::tensor_list grad_outputs) {\n\
    torch::Tensor d_value_graph = grad_outputs[0];\n\
    auto saved = ctx->get_saved_variables();\n";
                autoGradFunction += " int li = ctx->saved_data[\"li\"].toInt();\n\
        torch::Tensor offset_graph = global_offset_graph[2 * li + 1];\n\
        torch::Tensor columns_graph = global_columns_graph[2 * li + 1];\n";

                if (isColTile){
                    autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li + 1];\n\
        int segments = global_segments[2 * li + 1];\n";
                } else
                {
                    autoGradFunction += "unsupported\n";
                }
                autoGradFunction += "    torch::Tensor value_graph = saved[0]; // n x 1\n\
    torch::Tensor sds = value_graph * d_value_graph; // e x 1\n\
    torch::Tensor accum = node_spmv_backward_of_sddmm_nln(\n\
        offset_graph, columns_graph, sds, bounds, global_nrows, segments); // n x 1\n\
    torch::Tensor res = inplace_softmax_sddvv_mult(\n\
        accum, offset_graph, columns_graph, value_graph, bounds, global_nrows,\n\
        segments);\n\
    res = sds - res;\n\
    return {res, torch::Tensor()};\n\
  }\n\
};\n";
                kernelCallCode.addCode(autoGradFunction);
            }
            auto inGraphIndx = cNode->getInput(0)->getDataInfo()->getIndex();
            std::string tempForwardAggrCall = generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
            + "_AutoGrad::apply(" + cNode->getInput(0)->getName() +", " + std::to_string(inGraphIndx) + ");";
            model.getForward()->addCode(tempForwardAggrCall);
        } else if (cNode->getOp() == AGGREGATE_MUL_SUM_OP)
        {
            if (hasCOpt(cNode, SAMPLE_COPT)){
                if (encounteredAutograds.find("random_r_b") == encounteredAutograds.end())
                {
                    encounteredAutograds.insert("random_r_b");
                    std::string forwardRandCall = "    std::random_device rd;\n\
    std::mt19937 gen(rd());\n\
    std::uniform_int_distribution<> distrib(0, 100);\n\
    global_ra = distrib(gen);\n\
    global_rb = distrib(gen);\n";
                    model.getForward()->addCode(forwardRandCall);
                }
            }
            
            if (hasFFNEdgeUpdate)
            {
                bool isColTile = hasDOpt(cNode->getInput(1), COL_TILE_DOPT);
                if (encounteredAutograds.find(getKernelName(cNode)) == encounteredAutograds.end())
                {
                    encounteredAutograds.insert(getKernelName(cNode));
                    std::string autoGradFunction = ""
    "class " + getKernelName(cNode) + "_AutoGrad : public torch::autograd::Function<" + getKernelName(cNode) + "_AutoGrad> {\n\
    public:\n\
        static torch::Tensor forward(torch::autograd::AutogradContext *ctx,\n\
                                     torch::Tensor input_dense, torch::Tensor value_graph, int li) {\n\
            ctx->saved_data[\"li\"] = li;\n\
            torch::Tensor offset_graph = global_offset_graph[2 * li];\n\
            torch::Tensor columns_graph = global_columns_graph[2 * li];\n\
            ctx->save_for_backward(\n\
                {value_graph, input_dense});\n";
                if (isColTile){
                    autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li];\n\
            int segments = global_segments[2 * li];\n\
             return " + getKernelName(cNode) + "_call(input_dense, offset_graph, columns_graph,\n\
                                value_graph, bounds, segments);\n";
                } else
                {
                    autoGradFunction += "        return " + getKernelName(cNode) + "_call(input_dense, offset_graph, columns_graph,\n\
                                      value_graph);\n";
                }
                autoGradFunction += "    }\n\
    \n\
        static torch::autograd::tensor_list\n\
        backward(torch::autograd::AutogradContext *ctx,\n\
                 torch::autograd::tensor_list grad_outputs) {\n\
            torch::Tensor dZ = grad_outputs[0];\n\
            auto saved = ctx->get_saved_variables();\n\
            torch::Tensor value_graph = saved[0];\n\
            torch::Tensor X = saved[1];\n\
            int li = ctx->saved_data[\"li\"].toInt();\n\
            torch::Tensor offset_graph = global_offset_graph[2 * li + 1];\n\
            torch::Tensor columns_graph = global_columns_graph[2 * li + 1];";
                if (isColTile){
                    autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li];\n\
            int segments = global_segments[2 * li];\n\
            return {" + getKernelName(cNode) + "_call(dZ, offset_graph, columns_graph, value_graph,\n\
 bounds, segments),\n\
 edge_sddmm(dZ, X, offset_graph, columns_graph, value_graph, bounds,\n\
           global_nrows, segments),\n\
 torch::Tensor()};";
                } else
                {
                    // TODO add codegen for non-col tile
                    autoGradFunction += "\
            return {" + getKernelName(cNode) + "_call(dZ, offset_graph, columns_graph,\n\
                                       value_graph),\n\
edge_sddmm(dZ, X, offset_graph, columns_graph, value_graph, bounds,\n\
           global_nrows, 1), torch::Tensor()};\n";
                }
        autoGradFunction += "\
        }\n\
    };";
            kernelCallCode.addCode(autoGradFunction);
                }
                if (outOfLoop)
                {
                    // TODO later make this better
                    // TODO Check if the output name is res, if not then pass this along to the output
                    auto inGraphIndx = cNode->getInput(1)->getDataInfo()->getIndex();
                    std::string tempForwardAggrCall;
                    if (cNode->getOutput(0)->getName() == "res_n" || cNode->getOutput(0)->getName() == "t_iden_n")
                    {
                        tempForwardAggrCall =  "torch::Tensor t_iden_n = " + getKernelName(cNode)
                   + "_AutoGrad::apply(t_iden, 0);";
                        std::string aggrResStr = ", t_iden_n";
                        model.getCall()->addCode(aggrResStr);
                        std::string aggrResForward = ", torch::Tensor t_iden_n";
                        model.getForwardCallInternal()->addCode(aggrResForward);
                    } else
                    {
                        tempForwardAggrCall =  "t_iden = " + getKernelName(cNode)
                   + "_AutoGrad::apply(t_iden, 0);"; // TODO: Alyways do 0 (for now, since it'll be used for all scenarios)
                    }

                    model.getInv()->addCode(tempForwardAggrCall);
                } else
                {
                    auto inGraphIndx = cNode->getInput(1)->getDataInfo()->getIndex();

                    std::string tempForwardAggrCall = "    if (ep % mod_v == 0) {\n\
      " + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
                    + "_AutoGrad::apply(" + cNode->getInput(0)->getName() +", attn, 0);\n\
    } else {\n\
      " + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
                    + "_AutoGrad::apply(" + cNode->getInput(0)->getName() +", attn, " + std::to_string(inGraphIndx) + ");\n\
    }";
                    model.getForward()->addCode(tempForwardAggrCall);
                }
            } else
            {
                bool isColTile = hasDOpt(cNode->getInput(1), COL_TILE_DOPT);
                if (encounteredAutograds.find(getKernelName(cNode)) == encounteredAutograds.end())
                {
                    encounteredAutograds.insert(getKernelName(cNode));
                    std::string autoGradFunction = ""
  
    "class " + getKernelName(cNode) + "_AutoGrad : public torch::autograd::Function<" + getKernelName(cNode) + "_AutoGrad> {\n\
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
             return " + getKernelName(cNode) + "_call(input_dense, offset_graph, columns_graph,\n\
                                value_graph, bounds, segments);\n";
                } else
                {
                    autoGradFunction += "        return " + getKernelName(cNode) + "_call(input_dense, offset_graph, columns_graph,\n\
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
                    autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li + 1];\n\
            int segments = global_segments[2 * li + 1];\n\
            return {" + getKernelName(cNode) + "_call(input_dense, offset_graph, columns_graph, value_graph, bounds, segments), torch::Tensor()};";
                } else
                {
                    autoGradFunction += "\
            return {" + getKernelName(cNode) + "_call(input_dense, offset_graph, columns_graph,\n\
                                       value_graph), torch::Tensor()};\n";
                }
        autoGradFunction += "\
        }\n\
    };";
            kernelCallCode.addCode(autoGradFunction);
                }
                if (outOfLoop)
                {
                    std::string tempForwardAggrCall;
                    // std::cout << "Name: " << cNode->getOutput(0)->getName() << std::endl;
                    if (cNode->getOutput(0)->getName() == "res_n" || cNode->getOutput(0)->getName() == "t_iden_n")
                    {
                        tempForwardAggrCall =  "torch::Tensor t_iden_n = " + getKernelName(cNode)
                   + "_AutoGrad::apply(t_iden, 0);";
                        std::string aggrResStr = ", t_iden_n";
                        model.getCall()->addCode(aggrResStr);
                        std::string aggrResForward = ", torch::Tensor t_iden_n";
                        model.getForwardCallInternal()->addCode(aggrResForward);
                    } else
                    {
                        tempForwardAggrCall =  "t_iden = " + getKernelName(cNode)
                   + "_AutoGrad::apply(t_iden, 0);"; // TODO: Alyways do 0 (for now, since it'll be used for all scenarios)
                    }
                    model.getInv()->addCode(tempForwardAggrCall);

                    // std::string tempForwardAggrCall =  "t_iden = " + getKernelName(cNode)
                    // + "_AutoGrad::apply(t_iden, 0);";
                    // model.getInv()->addCode(tempForwardAggrCall);
                } else
                {
                    auto inGraphIndx = cNode->getInput(1)->getDataInfo()->getIndex();
                    std::string tempForwardAggrCall = "    if (ep % mod_v == 0) {\n\
      " + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
                    + "_AutoGrad::apply(" + cNode->getInput(0)->getName() +", 0);\n\
    } else {\n\
      " + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
                    + "_AutoGrad::apply(" + cNode->getInput(0)->getName() +", " + std::to_string(inGraphIndx) + ");\n\
    }";
                    model.getForward()->addCode(tempForwardAggrCall);
                }
            }
        } else if (cNode->getOp() == AGGREGATE_MUL_SUM_DIRECT)
        {
            bool isColTile = hasDOpt(cNode->getInput(1), COL_TILE_DOPT);

            if (outOfLoop)
            {
                auto inGraphIndx = cNode->getInput(1)->getDataInfo()->getIndex();
                std::string directAggrCall = "  torch::Tensor offset_graph_ones = global_offset_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
  torch::Tensor columns_graph_ones = global_columns_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
  torch::Tensor value_graph_ones = global_value_graph[2 * " + std::to_string(inGraphIndx) + "];\n";

                if (isColTile){
  
                    directAggrCall += "  torch::Tensor bounds_ones = global_bounds[2 * " + std::to_string(inGraphIndx) + "];\n\
  int segments_ones = global_segments[2 * " + std::to_string(inGraphIndx) + "];\n"
                    + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode) + "_call(" + cNode->getInput(0)->getName() + ", offset_graph_ones, columns_graph_ones,\n\
    value_graph_ones, bounds_ones, segments_ones);\n";
                } else
                {
                    directAggrCall += "  " + generateOutputString(cNode, outOfLoop) +" = " + getKernelName(cNode) + "_call(" + cNode->getInput(0)->getName() + ", offset_graph_ones, columns_graph_ones,\n\
    value_graph_ones);\n";
                }
                model.getInv()->addCode(directAggrCall);

            } else
            {
                auto inGraphIndx = cNode->getInput(1)->getDataInfo()->getIndex();
                std::string directAggrCall = "            torch::Tensor offset_graph_ones = global_offset_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
          torch::Tensor columns_graph_ones = global_columns_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
          torch::Tensor value_graph_ones = global_value_graph[2 * " + std::to_string(inGraphIndx) + "];\n";

                if (isColTile){
                    directAggrCall += "        torch::Tensor bounds_ones = global_bounds[2 * " + std::to_string(inGraphIndx) + "];\n\
        int segments_ones = global_segments[2 * " + std::to_string(inGraphIndx) + "];\n        "
                    + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode) + "_call(" + cNode->getInput(0)->getName() + ", offset_graph_ones, columns_graph_ones,\n\
                            value_graph_ones, bounds_ones, segments_ones);\n";
                } else
                {
                    directAggrCall += "        " + generateOutputString(cNode, outOfLoop) +" = " + getKernelName(cNode) + "_call(" + cNode->getInput(0)->getName() + ", offset_graph_ones, columns_graph_ones,\n\
                                  value_graph_ones);\n";
                }
                model.getForward()->addCode(directAggrCall);
            }
        } else if (cNode->getOp() == POWER_OP)
        {
            if (outOfLoop)
            {
                std::string powerCall = "   " + generateOutputString(cNode, outOfLoop) + " = torch::pow(" + cNode->getInput(0)->getName() + ", " + cNode->getParam(0) + ").detach();";
                model.getInv()->addCode(powerCall);
                // TODO: Temporary method to add kernel call
                std::string tempPassDegree = "," + cNode->getOutput(0)->getName();
                model.getCall()->addCode(tempPassDegree);

                std::string tempPassDegreeForward = ", torch::Tensor " + cNode->getOutput(0)->getName();
                model.getForwardCallInternal()->addCode(tempPassDegreeForward);
            } else
            {
                std::string powerCall = "        " + generateOutputString(cNode, outOfLoop) + " = torch::pow(" + cNode->getInput(0)->getName() + ", " + cNode->getParam(0) + ");";
                model.getForward()->addCode(powerCall);
            }

        } else if (cNode->getOp() == ROW_BROADCAST_OP)
        {
            if (outOfLoop)
            {
                std::string rbCall = "  t_iden = (" + cNode->getInput(0)->getName() + " * t_iden).detach();";
                model.getInv()->addCode(rbCall);
            } else
            {
                std::string rbCall = "        " + generateOutputString(cNode, outOfLoop) + " = " + cNode->getInput(0)->getName()
            + " * " + cNode->getInput(1)->getName() + ";";
                model.getForward()->addCode(rbCall);
            }

        } else if (cNode->getOp() == NON_LNR_OP_RELU)
        {
            std::string reluCall = "        " + generateOutputString(cNode, outOfLoop) + " = torch::relu(" + cNode->getInput(0)->getName() + ");";
            model.getForward()->addCode(reluCall);
        } else if (cNode->getOp() == NON_LNR_OP_LEAKY_RELU)
        {
            if (encounteredAutograds.find(getKernelName(cNode)) == encounteredAutograds.end())
            {
                encounteredAutograds.insert(getKernelName(cNode));
                std::string leakyReluInit = "     torch::nn::LeakyReLU leaky_relu(torch::nn::LeakyReLUOptions().negative_slope(0.2));";
                model.getForward()->addCode(leakyReluInit);
            }
            std::string leakyReluCall = "     " + generateOutputString(cNode, outOfLoop) + " = leaky_relu->forward(" + cNode->getInput(0)->getName() + ");";
            model.getForward()->addCode(leakyReluCall);
        } else if (cNode->getOp() == FFN_OP)
        {
            // TODO Check if input and output names are same. If they are use same, if not use something else
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
                inputSizes.push_back(cNode->getInput(1)->getDataInfo()->getDimRow());
                inputSizes.push_back(cNode->getInput(1)->getDataInfo()->getDimCol());

                // TODO add the inputs to the forward call based on the actual inputs
                std::string forwardCall = generateOutputString(cNode, outOfLoop) + " = fc" + std::to_string(fcCount) + "->forward(" + cNode->getInput(0)->getName() + ");";
                model.getForward()->addCode(forwardCall);
            } else
            {
                std::string inSize2 = "int size" + std::to_string(fcCount + 1);
                model.getInitCall()->addCode(inSize2);

                std::string fcDef = "torch::nn::Linear fc" + std::to_string(fcCount) + "{nullptr};";
                model.getDef()->addCode(fcDef);

                std::string fcInit = "fc" + std::to_string(fcCount) + " = register_module(\"fc"
                + std::to_string(fcCount) + "\", torch::nn::Linear(size" + std::to_string(fcCount)
                + ", size" + std::to_string(fcCount + 1) + "));";
                model.getInit()->addCode(fcInit);

                inputSizes.push_back(cNode->getInput(1)->getDataInfo()->getDimCol());

                // TODO add the inputs to the forward call based on the actual inputs
                std::string forwardCall = generateOutputString(cNode, outOfLoop) + " = fc" + std::to_string(fcCount) + "->forward(" + cNode->getInput(0)->getName() + ");";
                model.getForward()->addCode(forwardCall);
  
            }
            fcCount++;
        }  else if (cNode->getOp() == FFN_OP_EDGE)
        {
            std::string fcDef = "torch::nn::Linear efc" + std::to_string(fcEdgeCount) + "{nullptr};";
            model.getDef()->addCode(fcDef);

            std::string fcInit = "efc" + std::to_string(fcEdgeCount) + " = register_module(\"efc"
            + std::to_string(fcEdgeCount) + "\", torch::nn::Linear(size" + std::to_string(fcCount)
            + ", 1));";
            model.getInit()->addCode(fcInit);

            // TODO add the inputs to the forward call based on the actual inputs
            std::string forwardCall = generateOutputString(cNode, outOfLoop) + " = efc" + std::to_string(fcEdgeCount) + "->forward(" + cNode->getInput(0)->getName() + ");";
            model.getForward()->addCode(forwardCall);
            fcEdgeCount++;
        } else if (cNode->getOp() == SCALAR_ADD_EPS_MULTIPLY_OP)
        {
            std::string epDef = "torch::Tensor eps" + std::to_string(epCount) + "{nullptr};";
            model.getDef()->addCode(epDef);

            std::string epInit = "eps" + std::to_string(epCount) + " = register_parameter(\"eps"
            + std::to_string(epCount) + "\", torch::tensor({(float)" + cNode->getParam(0) + "}));";
            model.getInit()->addCode(epInit);

            // TODO add the inputs to the forward call based on the actual inputs
            std::string forwardCall = generateOutputString(cNode, outOfLoop) + " = (1 + eps" + std::to_string(epCount) + ") * " + cNode->getInput(0)->getName() + ";";
            if (outOfLoop)
            {
                model.getInv()->addCode(forwardCall);
            } else
            {
                model.getForward()->addCode(forwardCall);
            }
            epCount++;
        } else if (cNode->getOp() == ADD_OP)
        {
            std::string forwardCall = generateOutputString(cNode, outOfLoop) + " = " + cNode->getInput(0)->getName() + " + " + cNode->getInput(1)->getName() + ";";
            if (outOfLoop)
            {
                model.getInv()->addCode(forwardCall);
            } else
            {
                model.getForward()->addCode(forwardCall);
            }
        } else if (cNode->getOp() == ONES_OP)
        {
            auto outputInfo = cNode->getOutput(0)->getDataInfo();

            std::string rowDims = processDims(outputInfo->getDimRow());
            std::string colDims = processDims(outputInfo->getDimCol());

            if (outOfLoop)
            {
                // TODO eventually use a device specific function for this.
                std::string tempOptionsOnes = "    auto options_" + cNode->getOutput(0)->getName() +" = torch::TensorOptions()\n\
                       .dtype(torch::kFloat)\n\
                       .requires_grad(false)\n\
                       .device(torch::kCUDA, 0);";
                model.getInv()->addCode(tempOptionsOnes);

                // TODO add the inputs to the forward call based on the actual inputs
                std::string onesCall =  generateOutputString(cNode, outOfLoop) + " = torch::ones({" + rowDims
                + ", " + colDims + "}, options_" + cNode->getOutput(0)->getName() + ");";
                model.getInv()->addCode(onesCall);
            } else
            {
                // TODO eventually use a device specific function for this.
                std::string tempOptionsOnes = "    auto options_" + cNode->getOutput(0)->getName() +" = torch::TensorOptions()\n\
                       .dtype(torch::kFloat)\n\
                       .requires_grad(false)\n\
                       .device(torch::kCUDA, 0);";
                model.getForward()->addCode(tempOptionsOnes);

                // TODO add the inputs to the forward call based on the actual inputs
                std::string onesCall =  generateOutputString(cNode, outOfLoop) + " = torch::ones({" + rowDims
                + ", " + colDims + "}, options_" + cNode->getOutput(0)->getName() + ");";
                model.getForward()->addCode(onesCall);
            }
        }
    }

// TODO Put this in the common codegen? Doesn't seem to have any context specific content yet
    void generateCode(std::vector<CIRNode*>& program,
        std::vector<TransformEdge*>& transforms)
    {
        std::vector<int> inputSizes;
        int fcCount = 0;
        int fcEdgeCount = 0;
        int epCount = 0;
        bool hasFFNEdgeUpdate = false;

        // TODO add data transformations before data preparation.
        //  Should come from a middle end transformation.
        std::unordered_set<std::string> encounteredAutograds;
        for (int i = 0; i < program.size(); i++)
        {
            CIRNode* outNode = program[i];
            auto oNode = dynamic_cast<ComputeNode*>(outNode);
            if (oNode)
            {
                generateOpCode(oNode, fcCount, fcEdgeCount, epCount, true, hasFFNEdgeUpdate, encounteredAutograds, inputSizes, transforms);

                // Generate the transfer code after the load operation
                if (oNode->getOp() == LOAD_OP)
                {
                    this->dataPrep(program);
                }

            } else {
                std::string modelDef = "struct GALAGNN : torch::nn::Module {";
                model.getDef()->addCode(modelDef);
                std::string modelCall =  "GALAGNN(";
                model.getInitCall()->addCode(modelCall);

                // TODO generate this based on the program
                std::string tempFowradCallPre = "std::vector<torch::Tensor>\n\
forward(torch::Tensor t_iden";
                model.getForwardCallPre()->addCode(tempFowradCallPre);
                std::string tempFowradCallPost = ", int ep, int mod_v){\n";
                model.getForwardCallPost()->addCode(tempFowradCallPost);

                std::unordered_set<std::string> encounteredTensors;
                // std::string resInit = "torch::Tensor res = input_dense;";
                // model.getForward()->addCode(resInit);

                auto loopNode = dynamic_cast<TrainingLoopNode*>(outNode);

                std::string numIterCode = "int num_iters = " + std::to_string(loopNode->getIter()) + ";";
                preCode.addCode(numIterCode);

                for (int ix = 0; ix < loopNode->getLoopNodeNum(); ix++)
                {
                    CIRNode* inNode = loopNode->getNode(ix);
                    auto cNode = dynamic_cast<ComputeNode*>(inNode);
                    if (encounteredTensors.find(generateOutputString(cNode, false)) == encounteredTensors.end())
                    {
                        encounteredTensors.insert(generateOutputString(cNode, false));
                        std::string initTensor = "    torch::Tensor " + generateOutputString(cNode, false) + ";";
                        model.getForwardCallPost()->addCode(initTensor);
                    }

                    generateOpCode(cNode, fcCount, fcEdgeCount, epCount, false, hasFFNEdgeUpdate, encounteredAutograds, inputSizes, transforms);
                }
                CIRNode* inNode = loopNode->getNode(loopNode->getLoopNodeNum()-1);
                auto cNode = dynamic_cast<ComputeNode*>(inNode);
                // TODO Change this. (Remove and replace)
                std::string tempReturn = "return {" + cNode->getOutput(0)->getName() + "};";
                model.getForward()->addCode(tempReturn);

                std::string closeForward = "    }\n"
                                           "};";
                model.getForward()->addCode(closeForward);

                std::string closeInit = "   }";
                model.getInit()->addCode(closeInit);

                std::string closeInitCall = "){\n";
                model.getInitCall()->addCode(closeInitCall);

                // TODO Change the embedding sizes based on the

                std::string tempModelInit = "auto net = std::make_shared<GALAGNN>(";
                for (int ei = 0; ei < inputSizes.size(); ei++)
                {
                    tempModelInit += processDims(inputSizes[ei]);
                    if (ei < inputSizes.size() - 1)
                    {
                        tempModelInit += ", ";
                    }
                }
                tempModelInit += ");";
                model.getPreCall()->addCode(tempModelInit);

                std::string modelTransfer = "net->to(device);";
                model.getPreCall()->addCode(modelTransfer);

                if (loopNode->getOptimizer() == ADAM)
                {
                    std::string optmCode = "torch::optim::Adam optimizer(\n\
    net->parameters(), torch::optim::AdamOptions(" + std::to_string(loopNode->getLearningRate()) +").weight_decay(5e-4));\n";
                    model.getPreCall()->addCode(optmCode);
                } else
                {
                    std::cout << "Optimizer not supported." << std::endl;
                }

                int testStep =  loopNode->getTestStep();
                if (testStep > 1){
                    std::string initTrinStepsStr = " int mod_v = " + std::to_string(testStep) + ";\n";
                    model.getPreCall()->addCode(initTrinStepsStr);
                } else {
                    std::string initTrinStepsStr = " int mod_v = 1;\n";
                    model.getPreCall()->addCode(initTrinStepsStr);
                }

                std::string skipEpochsStr = " int skip_cache_warmup = 5;\n";
                model.getPreCall()->addCode(skipEpochsStr);

                std::string timingInitStr = " double start, end;\n\
  double start_train, end_train;\n\
  std::vector<double> times_arr, times_arr_train;\n";
                model.getPreCall()->addCode(timingInitStr);



                // TODO generate this using the test loop
    //             std::string tempTrainLoopPreCall = "for (size_t epoch = 1; epoch <= num_iters; ++epoch) {\n\
    // // Reset gradients.\n\
    // optimizer.zero_grad();\n\
    // torch::Tensor prediction =\n\
    //     net->forward(t_iden";

                std::string tempTrainLoopPreCall = " for (size_t epoch = 1; epoch <= num_iters; ++epoch) {\n\
    // Reset gradients.\n\
    optimizer.zero_grad();\n\
    // Execute the model on the input data.\n\
    cudaDeviceSynchronize();\n\
    start = get_time();\n\
    torch::Tensor prediction =\n\
        net->forward(t_iden";

                std::string tempTrainLoopPostCall = ", epoch, mod_v)[0];\n\
    cudaDeviceSynchronize();\n\
    end = get_time();\n\
    cudaDeviceSynchronize();\n\
    start_train = get_time();\n\
    torch::Tensor prediction_train = prediction.index({t_train_mask});\n\
    torch::Tensor labels_train = t_labs.index({t_train_mask});\n\
    auto criterion = torch::nn::CrossEntropyLoss();\n\
    torch::Tensor d_loss = criterion(prediction_train, labels_train);\n\
    d_loss.backward();\n\
    optimizer.step();\n\
    cudaDeviceSynchronize();\n\
    end_train = get_time();\n\
    if (epoch % mod_v == 0) {\n\
      torch::Tensor prediction_test = prediction.index({t_test_mask});\n\
      torch::Tensor labels_test = t_labs.index({t_test_mask});\n\
      auto [pred_val, pred_idx] = torch::max({prediction_test}, 1);\n\
      auto correct = torch::sum(pred_idx == labels_test);\n\
      std::cout << \"Epoch \" << epoch << \" Loss: \" << d_loss.item<val_t>()\n\
                << \" Accuracy: \"\n\
                << (correct.item<val_t>() * 100.0 / labels_test.sizes()[0])\n\
                << std::endl;\n\
    } else {\n\
      std::cout << \"Epoch \" << epoch << \" Loss: \" << d_loss.item<val_t>()\n\
                << std::endl;\n\
    }\n\
    if (epoch >= skip_cache_warmup) {\n\
      times_arr.push_back(end - start);\n\
      times_arr_train.push_back(end_train - start_train);\n\
    }\n\
  }";
                
//                 std::string tempTrainLoopPostCall = ")[0];\n\
//     torch::Tensor prediction_train = prediction.index({t_train_mask});\n\
//     torch::Tensor labels_train = t_labs.index({t_train_mask});\n\
//     auto criterion = torch::nn::CrossEntropyLoss();\n\
//     torch::Tensor d_loss = criterion(prediction_train, labels_train);\n\
//     d_loss.backward();\n\
//     optimizer.step();\n\
//     torch::Tensor prediction_test = prediction.index({t_test_mask});\n\
//     torch::Tensor labels_test = t_labs.index({t_test_mask});\n\
//     auto [pred_val, pred_idx] = torch::max({prediction_test}, 1);\n\
//     auto correct = torch::sum(pred_idx == labels_test);\n\
//     std::cout << \"Epoch \" << epoch << \" Loss: \" << d_loss.item<val_t>()\n\
//               << \" Accuracy: \"\n\
//               << (correct.item<val_t>() * 100.0 / labels_test.sizes()[0])\n\
//               << std::endl;\n\
// }";
                model.getPreCall()->addCode(tempTrainLoopPreCall);
                model.getPostCall()->addCode(tempTrainLoopPostCall);
            }
        }

        std::string printTimes = "  std::cout << \"Inference: \" << calc_mean(times_arr) << \",\"\n\
            << calc_std(times_arr) << std::endl;\n\
  std::cout << \"Train: \" << calc_mean(times_arr_train) << \",\"\n\
            << calc_std(times_arr_train) << std::endl;\n\
  std::cout << \"Total: \" << calc_mean(times_arr) + calc_mean(times_arr_train) << std::endl;\n";
        postCode.addCode(printTimes);

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
int global_classes;\n\
int global_emb_size;\n\
int global_ra;\n\
int global_rb;\n\
std::vector<int> global_segments;\n\
bool global_is_directed;\n\
\n\
std::vector<torch::Tensor> global_offset_graph;\n\
std::vector<torch::Tensor> global_columns_graph;\n\
std::vector<torch::Tensor> global_value_graph;\n\
std::vector<torch::Tensor> global_bounds;\n";

        importCode.addCode(tempStdCommon);

        std::string mainFuncCode  = "int main(int argc, char **argv) {\n\
  typedef typename SM::itype iT;\n\
  typedef typename SM::ntype nT;\n\
  typedef typename SM::vtype vT;\n\
\n\
  typedef typename DM::itype diT;\n\
  typedef typename DM::ntype dnT;\n\
  typedef typename DM::vtype dvT;\n\
  auto options_int_tile = \n\
    torch::TensorOptions().dtype(torch::kInt).requires_grad(false);\n\
  auto options_float_tile = \n\
    torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);\n";
        preCode.addCode(mainFuncCode);
    }

    void writeCode(std::vector<CIRNode*> &program,
        std::vector<RelationEdge*>& dependencies,
        std::vector<RelationEdge*>& associations,
        std::vector<TransformEdge*>& transforms)
    {
        // Kernel code - Architecture dependant
        // CMake (also has write for now?)
        initCMake();
        // Kernels
        initKernels(program);
        commonPerCode(program);

        generateCode(program, transforms);

        this->writeCode(cmakeCode, outStreamCMake);
        this->writeCode(importCode, outStreamModel);
        this->writeCode(kernelCode, outStreamModel);
        this->writeCode(kernelCallCode, outStreamModel);
        this->writeCode(*model.getDef(), outStreamModel);
        this->writeCode(*model.getInitCall(), outStreamModel, ", ", true, true);
        this->writeCode(*model.getInit(), outStreamModel);
        this->writeCode(*model.getForwardCallPre(), outStreamModel, "");
        this->writeCode(*model.getForwardCallInternal(), outStreamModel, "");
        this->writeCode(*model.getForwardCallPost(), outStreamModel);
        this->writeCode(*model.getForward(), outStreamModel);
        this->writeCode(preCode, outStreamModel);
        this->writeCode(*model.getInv(), outStreamModel);
        this->writeCode(*model.getPreCall(), outStreamModel, "");
        this->writeCode(*model.getCall(), outStreamModel, "");
        this->writeCode(*model.getPostCall(), outStreamModel);
        this->writeCode(postCode, outStreamModel);

        this->closeStream();
    }
};

#endif //GNN_ACCELERATION_LANGUAGE_COMMON_H
