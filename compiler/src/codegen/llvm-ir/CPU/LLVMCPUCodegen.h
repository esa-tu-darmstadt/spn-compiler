//
// Created by ls on 10/9/19.
//

#ifndef SPNC_LLVMCPUCODEGEN_H
#define SPNC_LLVMCPUCODEGEN_H

#include <unordered_map>
#include <graph-ir/GraphIRNode.h>
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

class LLVMCPUCodegen {

public:
    explicit LLVMCPUCodegen();

    void generateLLVMIR(IRGraph& graph);

private:
    LLVMContext context;
    IRBuilder<> builder;
    std::shared_ptr<Module> module;
    std::unordered_map<std::string, Value*> node2value;

};


#endif //SPNC_LLVMCPUCODEGEN_H
