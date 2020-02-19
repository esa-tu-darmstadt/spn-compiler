//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "MLIRCodeGen.h"
#include "MLIRBodyGen.h"

using namespace spnc;

MLIRCodeGen::MLIRCodeGen(spnc::ActionWithOutput<spnc::IRGraph>& _input,
                         const std::string& _kernelName,
                         std::shared_ptr<MLIRContext> _context) : ActionSingleInput<IRGraph, ModuleOp>{_input},
                                                                  context{_context}, kernelName{_kernelName},
                                                                  builder{context.get()} {
  module = std::make_unique<ModuleOp>(mlir::ModuleOp::create(builder.getUnknownLoc()));
}

ModuleOp& MLIRCodeGen::execute() {
  if (!cached) {
    generateMLIR(input.execute());
    cached = true;
  }
  return *module;
}

void MLIRCodeGen::generateMLIR(spnc::IRGraph& graph) {
  auto spnFunc = createSPNFunction(graph.inputs->size());
  auto& entryBlock = *spnFunc.addEntryBlock();
  assert(entryBlock.getNumArguments() == 1 && "Expecting a single argument for SPN function!");
  auto inputArg = entryBlock.getArgument(0);
  builder.setInsertionPointToStart(&entryBlock);
  for (auto& input : *graph.inputs) {
    auto inVar = builder.create<InputVarOp>(builder.getUnknownLoc(), inputArg, input->index());
    node2value[input.get()] = inVar;
  }
  MLIRBodyGen bodygen{&builder, &node2value};
  graph.rootNode->accept(bodygen, nullptr);
  auto rootVal = node2value[graph.rootNode.get()];
  builder.create<ReturnOp>(builder.getUnknownLoc(), rootVal);
  module->push_back(spnFunc);
}

mlir::FuncOp MLIRCodeGen::createSPNFunction(uint32_t numInputs) {
  Type elementType = builder.getIntegerType(32);
  Type evidenceType = RankedTensorType::get({numInputs}, elementType);
  auto func_type = builder.getFunctionType({evidenceType}, {builder.getF64Type()});
  return FuncOp::create(builder.getUnknownLoc(), kernelName + "_spn", func_type);
}
