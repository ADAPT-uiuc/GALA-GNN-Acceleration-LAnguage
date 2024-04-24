//
// Created by damitha on 4/23/24.
//
#ifndef GNN_ACCELERATION_LANGUAGE_COMMON_H
#define GNN_ACCELERATION_LANGUAGE_COMMON_H

#include <string>
#include <vector>
#include "../ir/compute.h"
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

class CodeGenerator{
private:
    GALAContext* context;
    std::ofstream outStream;
    std::vector<std::string> moreImports;
    std::vector<std::string> mainCode;
public:
    CodeGenerator(GALAContext* context, std::string* outputPath){
        this->context = context;
        this->openStream(outputPath);
    }
    ~CodeGenerator(){
        this->closeStream();
        this->moreImports.clear();
    }

    void generateCode (std::vector<ComputeNode> &program);

    GALAContext* getContext(){
        return this->context;
    }

    // Handle the stream to write to
    void openStream(std::string *outputPath){
        this->outStream = std::ofstream(*outputPath);
    }
    void closeStream(){
        this->outStream.close();
    }
    void writeLine(const std::string &str){
        this->outStream << str;
        this->outStream << "\n";
    }
    void write(const std::string &str){
        this->outStream << str;
    }

    void addImport(const std::string &str){
        this->moreImports.push_back(str);
    }
    void addCode(const std::string &str){
        this->mainCode.push_back(str);
    }

    // Initialization of the common code segments
    void initCode(){
        // Make the necessary imports etc. for the code
        writeLine("import torch");
        writeLine("from dgl.utils import expand_as_pair");
        writeLine("from dgl import function as fn");
        writeLine("import numpy as np");
        for (const auto &mImp: this->moreImports){
            writeLine(mImp);
        }
        writeLine("");
        writeLine("torch.ops.load_library(\"build/libgala_fn.so\")");
    }
    void writeCode(){
        // Make the necessary imports etc. for the code
        for (const auto &cde: this->mainCode){
            writeLine(cde);
        }
    }
};

#endif //GNN_ACCELERATION_LANGUAGE_COMMON_H
