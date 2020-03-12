//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <codegen/llvm-ir/CPU/body/CodeGenScalarBody.h>
#include "CodeGenSerialLoop.h"

using namespace spnc;

void CodeGenSerialLoop::emitLoop(Function& function, IRBuilder<>& builder, Value* lowerBound, Value* upperBound) {
  // Create a simple CFG for the loop comprising three main blocks.
  auto preheader = builder.GetInsertBlock();
  auto header = BasicBlock::Create(module.getContext(), "serial.loop.header", &function);
  auto body = BasicBlock::Create(module.getContext(), "serial.loop.body", &function);
  auto end = BasicBlock::Create(module.getContext(), "serial.loop.end", &function);
  builder.CreateBr(header);
  builder.SetInsertPoint(header);
  // Construct the PHI for the loop induction variable.
  auto phi = builder.CreatePHI(lowerBound->getType(), 2, "indVar");
  phi->addIncoming(lowerBound, preheader);
  auto comp = builder.CreateICmpULT(phi, upperBound, "serial.loop.comp");
  builder.CreateCondBr(comp, body, end);
  builder.SetInsertPoint(body);
  // Invoke the code-generation for the body.
  CodeGenScalarBody codegenBody{module, function, builder};
  auto incrVar = codegenBody.emitBody(graph, phi,
                                      getDefaultInputMap(function, builder),
                                      getDefaultOutputMap(function, builder));
  builder.CreateBr(header);
  phi->addIncoming(incrVar, body);
  builder.SetInsertPoint(end);
}

std::vector<Type*> CodeGenSerialLoop::constructInputArgumentTypes() {
  // Construct the single input argument, a pointer to a struct containing as many integers
  // as the SPN has features.
  auto intType = Type::getInt32Ty(module.getContext());
  auto structElements = std::vector<Type*>(graph.inputs().size(), intType);
  auto activationType = StructType::create(module.getContext(), structElements, "activation_t", false);
  auto activationPtrType = PointerType::get(activationType, 0);
  return std::vector<Type*>{activationPtrType};
}

std::vector<Type*> CodeGenSerialLoop::constructOutputArgumentTypes() {
  // Construct the single output argument, a pointer to double.
  return std::vector<Type*>{Type::getDoublePtrTy(module.getContext(), 0)};
}

