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
        LLVMCPUCodegen(ActionWithOutput<IRGraph>& _input, const std::string& _kernelName,
                std::shared_ptr<LLVMContext> _llvmContext);

        void generateLLVMIR(IRGraph& graph);

        llvm::Module& execute() override ;

    private:
        std::shared_ptr<LLVMContext> context;
        IRBuilder<> builder;
        std::unique_ptr<Module> module;
        std::string kernelName;
        std::unordered_map<std::string, Value*> node2value;
        bool cached = false;

    };
}




#endif //SPNC_LLVMCPUCODEGEN_H
