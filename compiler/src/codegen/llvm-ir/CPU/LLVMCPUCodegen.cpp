//
// Created by ls on 10/9/19.
//

#include <codegen/llvm-ir/CPU/loop/CodeGenSerialLoop.h>
#include "LLVMCPUCodegen.h"

namespace spnc {

    LLVMCPUCodegen::LLVMCPUCodegen(ActionWithOutput<IRGraph>& _input, const std::string& _kernelName)
      : ActionSingleInput<IRGraph, llvm::Module>(_input), builder{context}, kernelName{_kernelName} {
      module = std::make_unique<Module>("spn-llvm", context);
    }

    void LLVMCPUCodegen::generateLLVMIR(IRGraph &graph) {
      CodeGenSerialLoop codegenLoop{*module, graph};
      auto int64Ty = IntegerType::getInt64Ty(context);
      std::vector<Type*> argTypes;
      argTypes.push_back(int64Ty);
      auto inputArgs = codegenLoop.constructInputArgumentTypes();
      argTypes.insert(argTypes.end(), inputArgs.begin(), inputArgs.end());
      auto outputArgs = codegenLoop.constructOutputArgumentTypes();
      argTypes.insert(argTypes.end(), outputArgs.begin(), outputArgs.end());
      auto functionType = FunctionType::get(Type::getVoidTy(context), argTypes, false);
      auto function = Function::Create(functionType, Function::ExternalLinkage, kernelName, module.get());
      auto bb = BasicBlock::Create(context, "main", function);
      builder.SetInsertPoint(bb);
      codegenLoop.emitLoop(*function, builder, ConstantInt::get(int64Ty, 0), function->arg_begin());
      builder.CreateRetVoid();
      module->dump();
    }

    llvm::Module & LLVMCPUCodegen::execute() {
      if(!cached){
        generateLLVMIR(input.execute());
        cached = true;
      }
      return *module;
    }

}

