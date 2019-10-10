//
// Created by ls on 10/9/19.
//

#ifndef SPNC_LLVMCODEGEN_H
#define SPNC_LLVMCODEGEN_H

#include <unordered_map>
#include <graph-ir/GraphIRNode.h>
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

class LLVMCodegen {

public:
    explicit LLVMCodegen();

    void generateLLVMIR(IRGraph& graph);

private:
    LLVMContext context;
    IRBuilder<> builder;
    std::unique_ptr<Module> module;
    std::unordered_map<std::string, Value*> node2value;


};


#endif //SPNC_LLVMCODEGEN_H
