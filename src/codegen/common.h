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
    // Code for the model use
    Code modelTraining;
    Code modelValidation;
    Code modelTesting;

    // Code for the model defition
    // The code in the initialization section of the model + training invariant code
    Code modelInit;
    // Model forward
    Code modelForward;

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

    // Init
    Code* getInit()
    {
        return &this->modelInit;
    }

    // Def
    Code* getUse()
    {
        return &this->modelUse;
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
    Code importCode;
    Code preCode;
    Model model; // TODO: Assume a single model for now
    Code postCode;
    Code kernelCode;

    std::vector<std::string> generatedFunctions;

    std::ofstream outStreamModel;
    std::ofstream outStreamCMake;

public:
    CodeGenerator(GALAContext* context, std::string& outputPath)
    {
        this->context = context;
        this->openStream(outputPath);
    }

    virtual void generateCode(std::vector<ComputeNode*>& program)
    {
    };

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

    void writeCode(Code &code, std::ofstream &outStream, const std::string &end = "\n"){
        for (int ix = 0; ix < code.getNum(); ix++){
            auto codeLine = code.atLine(ix);
            outStream << *codeLine << end;
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
    virtual void initKernels()
    {
    };

    void writeCode(std::vector<CIRNode*> program)
    {
        // Kernel code - Architecture dependant
        // CMake (also has write for now?)
        initCMake();
        // Kernels
        initKernels();

        this->writeCode(cmakeCode, outStreamCMake);
        this->writeCode(preCode, outStreamCMake);
        this->closeStream();

        //        // Kernel binding to PyTorch -- Architecture independent
        //        for (const auto &kernel: this->kernelCode){
        //            for (int ib = 0; ib < kernel->getNumBind(); ib++){
        //                auto bindCodeLine = kernel->getBind(ib);
        //                writePyLine(bindCodeLine);
        //            }
        //        }
        //
        //        // Model Def / Forward -- Prior to the main Python function
        //        for (const auto &model: this->modelCode){
        //            std::string modelNameStr("class " + *model->getName() + "(torch.nn.Module)");
        //            auto modelNameLine = PyCodeLine(modelNameStr);
        //            writePyLine(&modelNameLine);
        //
        //            for (int id = 0; id < model->getNumDef(); id++){
        //                auto defCodeLine = model->atDef(id);
        //                writePyLine(defCodeLine);
        //            }
        //
        //            for (int ig = 0; ig < model->getNumForward(); ig++){
        //                auto forCodeLine = model->atForward(ig);
        //                writePyLine(forCodeLine);
        //            }
        //        }
        //
        //        // Code before model def and use (Ex - Load graph)
        //        for (const auto &cde: this->preCode){
        //            writePyLine(cde);
        //        }
        //
        //        // Model Init / Train / Validation / Test Code
        //        // Model Def / Forward -- Prior to the main Python function
        //        for (const auto &model: this->modelCode){
        //            for (int it = 0; it < model->getNumTraining(); it++){
        //                auto trainCodeLine = model->atTraining(it);
        //                writePyLine(trainCodeLine);
        //            }
        //
        //            for (int iv = 0; iv < model->getNumTraining(); iv++){
        //                auto validCodeLine = model->atValidation(iv);
        //                writePyLine(validCodeLine);
        //            }
        //
        //            for (int ig = 0; ig < model->getNumTesting(); ig++){
        //                auto testCodeLine = model->atTesting(ig);
        //                writePyLine(testCodeLine);
        //            }
        //        }
        //
        //        // Code after training (Ex - Evaluate correctness of results)
        //        for (const auto &cde: this->postCode){
        //            writePyLine(cde);
        //        }
    }
};

#endif //GNN_ACCELERATION_LANGUAGE_COMMON_H
