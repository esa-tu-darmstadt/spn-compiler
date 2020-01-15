//
// Created by ls on 10/9/19.
//

#ifndef SPNC_LLVMCPUCODEGEN_H
#define SPNC_LLVMCPUCODEGEN_H

#include <unordered_map>
#include <graph-ir/GraphIRNode.h>
#include <driver/Actions.h>
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace spnc {

    class LLVMCPUCodegen : public ActionSingleInput<IRGraph, llvm::Module> {

    public:
        explicit LLVMCPUCodegen(ActionWithOutput<IRGraph>& _input);

        void generateLLVMIR(IRGraph& graph);

        llvm::Module& execute() override ;

    private:
        LLVMContext context;
        IRBuilder<> builder;
        std::unique_ptr<Module> module;
        std::unordered_map<std::string, Value*> node2value;
        bool cached = false;

    };
}




#endif //SPNC_LLVMCPUCODEGEN_H
