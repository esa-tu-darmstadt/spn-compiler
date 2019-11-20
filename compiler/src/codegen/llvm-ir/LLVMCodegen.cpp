//
// Created by ls on 10/9/19.
//

#include <codegen/llvm-ir/operations/CodeGenOperations.h>
#include "LLVMCodegen.h"

LLVMCodegen::LLVMCodegen() : builder{context} {
    module = std::make_shared<Module>("spn-llvm", context);
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
    InputVarValueMap inputVarMap = [function](size_t index, IRBuilder<>& builder, LLVMContext& context){
        auto inputArg = function->arg_begin();
        assert(inputArg->getType()->isPointerTy() && "Expecting input to be a pointer type!");
        assert(((PointerType*)inputArg->getType())->getElementType()->isAggregateType()
        && "Expecting input to be a struct!");
        auto gep = builder.CreateGEP(inputArg,
                {ConstantInt::get(IntegerType::get(context, 32), 0),
                ConstantInt::get(IntegerType::get(context, 32), index)}, "gep_input");
        return builder.CreateLoad(gep, "input_value");
    };
    CodeGenOperations codeGenOperations{*module, *function, builder, inputVarMap};
    graph.rootNode->accept(codeGenOperations, nullptr);
    builder.CreateRetVoid();
    module->dump();
}