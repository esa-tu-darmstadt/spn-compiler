//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <codegen/llvm-ir/CPU/loop/CodeGenSerialLoop.h>
#include "LLVMCPUCodegen.h"

using namespace spnc;

LLVMCPUCodegen::LLVMCPUCodegen(ActionWithOutput<IRGraph>& _input, std::string _kernelName,
                               std::shared_ptr<LLVMContext> _llvmContext)
    : ActionSingleInput<IRGraph, llvm::Module>(_input),
      context{std::move(_llvmContext)}, builder{*context}, kernelName{std::move(_kernelName)} {
  module = std::make_unique<Module>("spn-llvm", *context);
}

void LLVMCPUCodegen::generateLLVMIR(IRGraph& graph) {
  CodeGenSerialLoop codegenLoop{*module, graph};
  auto int64Ty = IntegerType::getInt64Ty(*context);
  std::vector<Type*> argTypes;
  // Argument indicating the number of queries in the batch.
  argTypes.push_back(int64Ty);
  // Get the correct input- and output-types from the loop code-generation.
  auto inputArgs = codegenLoop.constructInputArgumentTypes();
  argTypes.insert(argTypes.end(), inputArgs.begin(), inputArgs.end());
  auto outputArgs = codegenLoop.constructOutputArgumentTypes();
  argTypes.insert(argTypes.end(), outputArgs.begin(), outputArgs.end());
  // Create the top-level function with the specified name.
  auto functionType = FunctionType::get(Type::getVoidTy(*context), argTypes, false);
  auto function = Function::Create(functionType, Function::ExternalLinkage, kernelName, module.get());
  // Create a single basic block and invoke the loop code-generation.
  auto bb = BasicBlock::Create(*context, "main", function);
  builder.SetInsertPoint(bb);
  codegenLoop.emitLoop(*function, builder, ConstantInt::get(int64Ty, 0), function->arg_begin());
  builder.CreateRetVoid();
}

llvm::Module& LLVMCPUCodegen::execute() {
  if (!cached) {
    generateLLVMIR(input.execute());
    cached = true;
  }
  return *module;
}


