//
// Created by lukas on 20.11.19.
//
#include <codegen/llvm-ir/CPU/body/CodeGenScalarBody.h>
#include "CodeGenSerialLoop.h"

CodeGenSerialLoop::CodeGenSerialLoop(Module& m, IRGraph& g) : CodeGenLoop(m, g) {}

void CodeGenSerialLoop::emitLoop(Function &function, IRBuilder<> &builder, Value *lowerBound, Value *upperBound) {
    auto preheader = builder.GetInsertBlock();
    auto header = BasicBlock::Create(module.getContext(), "serial.loop.header", &function);
    auto body = BasicBlock::Create(module.getContext(), "serial.loop.body", &function);
    auto end = BasicBlock::Create(module.getContext(), "serial.loop.end", &function);
    builder.CreateBr(header);
    builder.SetInsertPoint(header);
    auto phi = builder.CreatePHI(lowerBound->getType(), 2, "indVar");
    phi->addIncoming(lowerBound, preheader);
    auto comp = builder.CreateICmpULT(phi, upperBound, "serial.loop.comp");
    builder.CreateCondBr(comp, body, end);
    builder.SetInsertPoint(body);
    CodeGenScalarBody codegenBody{module, function, builder};
    auto incrVar = codegenBody.emitBody(graph, phi,
            getDefaultInputMap(function, builder, module.getContext()),
            getDefaultOutputMap(function, builder));
    builder.CreateBr(header);
    phi->addIncoming(incrVar, body);
    builder.SetInsertPoint(end);
}

std::vector<Type*> CodeGenSerialLoop::constructInputArgumentTypes() {
    auto intType = Type::getInt32Ty(module.getContext());
    auto structElements = std::vector<Type*>(graph.inputs->size(), intType);
    auto activationType = StructType::create(module.getContext(), structElements, "activation_t", false);
    auto activationPtrType = PointerType::get(activationType, 0);
    return std::vector<Type*>{activationPtrType};
}

std::vector<Type*> CodeGenSerialLoop::constructOutputArgumentTypes() {
    return std::vector<Type*>{Type::getDoublePtrTy(module.getContext(), 0)};
}