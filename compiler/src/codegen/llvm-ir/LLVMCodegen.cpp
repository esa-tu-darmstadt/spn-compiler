//
// Created by ls on 10/9/19.
//

#include "LLVMCodegen.h"

LLVMCodegen::LLVMCodegen() : builder{context} {
    module = std::make_unique<Module>("spn-llvm", context);
}

void LLVMCodegen::generateLLVMIR(IRGraph &graph) {
    auto intType = Type::getInt32Ty(context);
    auto structElements = std::vector<Type*>(graph.inputs->size(), intType);
    auto activationType = StructType::create(context, structElements, "activation_t", false);
    auto activationPtrType = PointerType::get(activationType, 0);
    std::vector<Type*> argTypes{activationPtrType, Type::getDoublePtrTy(context, 0), Type::getInt64Ty(context)};
    auto functionType = FunctionType::get(Type::getVoidTy(context), argTypes, false);
    auto function = Function::Create(functionType, Function::ExternalLinkage, "spn_element", module.get());
    auto bb = BasicBlock::Create(context, "main", function);
    builder.SetInsertPoint(bb);
    module->dump();
}