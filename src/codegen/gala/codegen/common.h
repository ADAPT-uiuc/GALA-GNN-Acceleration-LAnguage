//
// Created by damitha on 4/23/24.
//
#ifndef GNN_ACCELERATION_LANGUAGE_COMMON_H
#define GNN_ACCELERATION_LANGUAGE_COMMON_H

#include <string>
#include <vector>
#include "../ir/compute.h"
#include <fstream>
#include <iostream>

enum Device{
    CPU_DEVICE,
    GPU_DEVICE
};

enum Environment{
    SINGLE_NODE_SINGLE,
    SINGLE_NODE_MULTI,
    MULTI_NODE_SINGLE,
    MULTI_NODE_MULTI
};

class GALAContext{
private:
    Device device;
    Environment env;
public:
    GALAContext(Device dev, Environment env){
        this->device = dev;
        this->env = env;
    }

    Device getDevice(){
        return this->device;
    }
    Environment getEnv(){
        return this->env;
    }
};

class PyCodeLine{
private:
    int indentLevel;
    std::string codeLine;
public:
    PyCodeLine(std::string &code){
        this->indentLevel = 0;
        this->codeLine = code;
    }
    PyCodeLine(int level, std::string &code){
        this->indentLevel = level;
        this->codeLine = code;
    }

    int getIndent(){
        return this->indentLevel;
    }
    std::string* getCode(){
        return &this->codeLine;
    }
};

// TODO -- For now both CmakeCode and KernelCode have the same operations
//  Separate this out. CMake is common for all, but the kernel code would also have
//  a PyTorch linking component
class BaseKernelCode{
private:
    std::vector<std::string> codeLines;
public:
    BaseKernelCode(){};
    int getNum(){
        return this->codeLines.size();
    }
    void addCode(std::string &newCode){
        this->codeLines.push_back(newCode);
    }
    std::string* atLine(int ix){
        return &(this->codeLines.at(ix));
    }
};

class CMakeCode : public BaseKernelCode{
};

class KernelCode : public BaseKernelCode{
private:
    std::string name;
    std::vector<PyCodeLine*> bindTorch;
public:
    KernelCode(std::string &name):BaseKernelCode(){
        this->name = name;
    };
    std::string* getName(){
        return &this->name;
    }
    int getNumBind(){
        return this->bindTorch.size();
    }
    void addBind(PyCodeLine* newCode){
        this->bindTorch.push_back(newCode);
    }
    PyCodeLine* getBind(int ix){
        return this->bindTorch.at(ix);
    }
};


// PyTorch mdoel code
class ModelCode{
private:
    // ID for the model
    std::string modelName;
    // The code in the initialization section of the model + training invariant code
    std::vector<PyCodeLine*> modelInit;
    // Model component definition (The init of a Torch model)
    std::vector<PyCodeLine*> modelDef;
    // Model forward
    std::vector<PyCodeLine*> modelForward;

    std::vector<PyCodeLine*> modelTraining;
    std::vector<PyCodeLine*> modelValidation;
    std::vector<PyCodeLine*> modelTesting;
public:
    ModelCode(std::string &name){
        this->modelName = name;
    }
    // TODO this is at the code generation phase so you don't need to clear / remove stuff
    //  Those should already have been decided in previous passes
    std::string* getName(){
        return &this->modelName;
    }
    // Init
    void addInit(PyCodeLine* newCode){
        this->modelInit.push_back(newCode);
    }
    int getNumInit(){
        return (int)this->modelInit.size();
    }
    PyCodeLine* atInit(int ix){
        return this->modelInit.at(ix);
    }
    // Def
    void addDef(PyCodeLine* &newCode){
        this->modelDef.push_back(newCode);
    }
    int getNumDef(){
        return (int)this->modelDef.size();
    }
    PyCodeLine* atDef(int ix){
        return this->modelDef.at(ix);
    }
    // Forward
    void addForward(PyCodeLine* &newCode){
        this->modelForward.push_back(newCode);
    }
    int getNumForward(){
        return (int)this->modelForward.size();
    }
    PyCodeLine* atForward(int ix){
        return this->modelForward.at(ix);
    }
    // Training
    void addTraining(PyCodeLine* &newCode){
        this->modelTraining.push_back(newCode);
    }
    int getNumTraining(){
        return (int)this->modelTraining.size();
    }
    PyCodeLine* atTraining(int ix){
        return this->modelTraining.at(ix);
    }
    // Validation
    void addValidation(PyCodeLine* &newCode){
        this->modelValidation.push_back(newCode);
    }
    int getNumValidation(){
        return (int)this->modelValidation.size();
    }
    PyCodeLine* atValidation(int ix){
        return this->modelValidation.at(ix);
    }
    // Testing
    void addTesting(PyCodeLine* &newCode){
        this->modelTesting.push_back(newCode);
    }
    int getNumTesting(){
        return (int)this->modelTesting.size();
    }
    PyCodeLine* atTesting(int ix){
        return this->modelTesting.at(ix);
    }
};

// TODO You're generating code for Python. So you NEED to identify where to make any indents
class CodeGenerator{
private:
    GALAContext* context;

    std::ofstream outStreamModel;

    std::vector<PyCodeLine*> moreImports;
    std::vector<PyCodeLine*> preCode;
    std::vector<ModelCode*> modelCode;
    std::vector<PyCodeLine*> postCode;

    // TODO This still needs to be here since you have the binding code
    std::vector<KernelCode*> kernelCode;

    std::vector<std::string> generatedFunctions;
protected:
    std::ofstream outStreamCMake;
    std::ofstream outStreamKernel;
public:
    CodeGenerator(GALAContext* context, std::string* outputPath){
        this->context = context;
        this->openStream(outputPath);
    }
    ~CodeGenerator(){
        this->closeStream();
        this->moreImports.clear();
    }

    virtual void generateCode (std::vector<ComputeNode*> &program);

    GALAContext* getContext(){
        return this->context;
    }

    // Handle the stream to write to
    void openStream(std::string *outputPath){
        std::string mainPath = *outputPath + "gala.py";
        this->outStreamModel = std::ofstream(mainPath);

        std::string cmakePath = *outputPath + "CMakeLists.txt";
        this->outStreamModel = std::ofstream(mainPath);

        std::string kernelPath = *outputPath + "gala.cpp";
        this->outStreamModel = std::ofstream(mainPath);
    }
    void closeStream(){
        this->outStreamModel.close();
        this->outStreamCMake.close();
        this->outStreamKernel.close();
    }
    void writePyLine(PyCodeLine* pyCode, const std::string &end = "\n"){
        for (int i = 0; i < pyCode->getIndent(); i++){
            this->outStreamModel << "\t";
        }
        this->outStreamModel << pyCode->getCode();
        this->outStreamModel << end;
    }

//    void writeCMake(CMakeCode* cmCode, const std::string &end = "\n"){
//        for (int ic = 0; ic < cmCode->getNum(); ic++){
//            auto codeLine = cmCode->atLine(ic);
//            this->outStreamCMake << *codeLine << end;
//        }
//    }

//    void writeKernel(KernelCode* kCode, const std::string &end = "\n"){
//        for (int ik = 0; ik < kCode->getNum(); ik++){
//            auto codeLine = kCode->atLine(ik);
//            this->outStreamKernel << *codeLine << end;
//        }
//    }


    void addImport(PyCodeLine* pyCode){
        this->moreImports.push_back(pyCode);
    }
    void addPreCode(PyCodeLine* pyCode){
        this->preCode.push_back(pyCode);
    }
    void addPostCode(PyCodeLine* pyCode){
        this->postCode.push_back(pyCode);
    }

    // Separate function so it can be extended in the architecture specific components
    virtual void initCMake();

    // Separate function so it can be extended in the architecture specific components
    virtual void initKernels();

    void writeCode(){
        // Kernel code - Architecture dependant
        // CMake
        initCMake();
        // Kernels
        initKernels();

        // PyTorch
        std::string im1("import torch");
        auto pl1 = PyCodeLine(im1);
        writePyLine(&pl1);
        std::string im2("import torch");
        auto pl2 = PyCodeLine(im2);
        writePyLine(&pl2);
        std::string im3("from dgl.utils import expand_as_pair");
        auto pl3 = PyCodeLine(im3);
        writePyLine(&pl3);
        std::string im4("from dgl import function as fn");
        auto pl4 = PyCodeLine(im4);
        writePyLine(&pl4);
        std::string im5("import numpy as np");
        auto pl5 = PyCodeLine(im5);
        writePyLine(&pl5);

        for (const auto &mImp: this->moreImports){
            writePyLine(mImp);
        }
        std::string im6("torch.ops.load_library(\"build/libgala_fn.so\")");
        auto pl6 = PyCodeLine(im6);
        writePyLine(&pl6);

        // Kernel binding to PyTorch -- Architecture independent
        for (const auto &kernel: this->kernelCode){
            for (int ib = 0; ib < kernel->getNumBind(); ib++){
                auto bindCodeLine = kernel->getBind(ib);
                writePyLine(bindCodeLine);
            }
        }

        // Model Def / Forward -- Prior to the main Python function
        for (const auto &model: this->modelCode){
            std::string modelNameStr("class " + *model->getName() + "(torch.nn.Module)");
            auto modelNameLine = PyCodeLine(modelNameStr);
            writePyLine(&modelNameLine);

            for (int id = 0; id < model->getNumDef(); id++){
                auto defCodeLine = model->atDef(id);
                writePyLine(defCodeLine);
            }

            for (int ig = 0; ig < model->getNumForward(); ig++){
                auto forCodeLine = model->atForward(ig);
                writePyLine(forCodeLine);
            }
        }

        // Code before model def and use (Ex - Load graph)
        for (const auto &cde: this->preCode){
            writePyLine(cde);
        }

        // Model Init / Train / Validation / Test Code
        // Model Def / Forward -- Prior to the main Python function
        for (const auto &model: this->modelCode){
            for (int it = 0; it < model->getNumTraining(); it++){
                auto trainCodeLine = model->atTraining(it);
                writePyLine(trainCodeLine);
            }

            for (int iv = 0; iv < model->getNumTraining(); iv++){
                auto validCodeLine = model->atValidation(iv);
                writePyLine(validCodeLine);
            }

            for (int ig = 0; ig < model->getNumTesting(); ig++){
                auto testCodeLine = model->atTesting(ig);
                writePyLine(testCodeLine);
            }
        }

        // Code after training (Ex - Evaluate correctness of results)
        for (const auto &cde: this->postCode){
            writePyLine(cde);
        }
    }
};

#endif //GNN_ACCELERATION_LANGUAGE_COMMON_H
