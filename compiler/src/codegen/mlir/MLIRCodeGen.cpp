//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "MLIRCodeGen.h"
#include "MLIRBodyGen.h"
#include <mlir/Analysis/Verifier.h>

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
  std::string bodyFuncName = kernelName + "_body";
  generateSPNBody(graph, bodyFuncName);
  generateSPNToplevel(graph, bodyFuncName);
  auto verificationResult = ::mlir::verify(module->getOperation());
  if (verificationResult.value == LogicalResult::Failure) {
    throw std::runtime_error("Verification of the generated MLIR module failed!");
  }
}

void MLIRCodeGen::generateSPNToplevel(spnc::IRGraph& graph, const std::string& bodyFuncName) {
  auto restore = builder.saveInsertionPoint();
  Type elementType = builder.getIntegerType(32);
  Type evidenceType = RankedTensorType::get({(int) graph.inputs().size()}, elementType);
  auto func_type = builder.getFunctionType({evidenceType}, {builder.getF64Type()});
  auto func = FuncOp::create(builder.getUnknownLoc(), kernelName, func_type);
  auto& entryBlock = *func.addEntryBlock();
  assert(entryBlock.getNumArguments() == 1 && "Expecting a single argument for SPN function!");
  auto inputArg = entryBlock.getArgument(0);
  builder.setInsertionPointToStart(&entryBlock);
  auto query = builder.create<SPNSingleQueryOp>(builder.getUnknownLoc(), inputArg, bodyFuncName);
  builder.create<ReturnOp>(builder.getUnknownLoc(), query);
  module->push_back(func);
  builder.restoreInsertionPoint(restore);
}

void MLIRCodeGen::generateSPNBody(spnc::IRGraph& graph, const std::string& funcName) {
  auto restore = builder.saveInsertionPoint();
  auto spnFunc = createSPNFunction(graph.inputs().size(), funcName);
  auto& entryBlock = *spnFunc.addEntryBlock();
  assert(entryBlock.getNumArguments() == 1 && "Expecting a single argument for SPN function!");
  auto inputArg = entryBlock.getArgument(0);
  builder.setInsertionPointToStart(&entryBlock);
  for (auto input : graph.inputs()) {
    auto inVar = builder.create<InputVarOp>(builder.getUnknownLoc(), inputArg, input->index());
    node2value[input->id()] = inVar;
  }
  MLIRBodyGen bodygen{&builder, &node2value};
  graph.rootNode()->accept(bodygen, nullptr);
  auto rootVal = node2value[graph.rootNode()->id()];
  builder.create<ReturnOp>(builder.getUnknownLoc(), rootVal);
  module->push_back(spnFunc);
  builder.restoreInsertionPoint(restore);
}

mlir::FuncOp MLIRCodeGen::createSPNFunction(uint32_t numInputs, const std::string& funcName) {
  Type elementType = builder.getIntegerType(32);
  Type evidenceType = RankedTensorType::get({numInputs}, elementType);
  auto func_type = builder.getFunctionType({evidenceType}, {builder.getF64Type()});
  return FuncOp::create(builder.getUnknownLoc(), funcName, func_type);
}
