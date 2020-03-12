//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "MLIRCodeGen.h"
#include "MLIRBodyGen.h"
#include <mlir/Analysis/Verifier.h>
#include <util/Logging.h>

using namespace spnc;

MLIRCodeGen::MLIRCodeGen(spnc::ActionWithOutput<spnc::IRGraph>& _input,
                         const std::string& _kernelName,
                         std::shared_ptr<MLIRContext> _context) : ActionSingleInput<IRGraph, ModuleOp>{_input},
                                                                  context{std::move(_context)}, kernelName{_kernelName},
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
  // Generate two function: One containing the SPN graph itself, suffixed with "_body"
  // and one top-level function triggering computation for a single query or a batch
  // of queries.
  std::string bodyFuncName = kernelName + "_body";
  generateSPNBody(graph, bodyFuncName);
  generateSPNToplevel(graph, bodyFuncName);
  // Verify the generated MLIR module.
  auto verificationResult = ::mlir::verify(module->getOperation());
  if (verificationResult.value == LogicalResult::Failure) {
    SPNC_FATAL_ERROR("Verification of the generated MLIR module failed!");
  }
  module->dump();
}

void MLIRCodeGen::generateSPNToplevel(spnc::IRGraph& graph, const std::string& bodyFuncName) {
  auto restore = builder.saveInsertionPoint();
  // Input is a single Tensor with as many elements as the SPN has features.
  Type elementType = builder.getIntegerType(32);
  Type evidenceType = RankedTensorType::get({(int) graph.inputs().size()}, elementType);
  // Create a function taking a single Tensor and returning the probability value as 64-bit float.
  auto func_type = builder.getFunctionType({evidenceType}, builder.getF64Type());
  auto func = FuncOp::create(builder.getUnknownLoc(), kernelName, func_type);
  auto& entryBlock = *func.addEntryBlock();
  assert(entryBlock.getNumArguments() == 1 && "Expecting two arguments for SPN function!");
  auto inputArg = entryBlock.getArgument(0);
  builder.setInsertionPointToStart(&entryBlock);
  // Trigger the single query using the corresponding operation from the SPN dialect.
  auto query = builder.create<SPNSingleQueryOp>(builder.getUnknownLoc(), inputArg, bodyFuncName);
  // Return the result.
  builder.create<ReturnOp>(builder.getUnknownLoc(), query);
  module->push_back(func);
  builder.restoreInsertionPoint(restore);
}

void MLIRCodeGen::generateSPNBody(spnc::IRGraph& graph, const std::string& funcName) {
  auto restore = builder.saveInsertionPoint();
  auto spnFunc = createSPNFunction(graph.inputs().size(), funcName);
  auto& entryBlock = *spnFunc.addEntryBlock();
  assert(entryBlock.getNumArguments() == graph.inputs().size()
             && "Expecting matching number of arguments for SPN function!");
  builder.setInsertionPointToStart(&entryBlock);
  // Establish a mapping from the i'th block argument to the feature at index i.
  for (auto input : graph.inputs()) {
    int index = input->index();
    auto inVar = builder.create<InputVarOp>(builder.getUnknownLoc(), entryBlock.getArgument(index), index);
    node2value[input->id()] = inVar;
  }
  // Invoke the code-generation for the graph itself.
  MLIRBodyGen bodygen{&builder, &node2value};
  graph.rootNode()->accept(bodygen, nullptr);
  auto rootVal = node2value[graph.rootNode()->id()];
  // Return the value computed by the operation generated for the root node.
  builder.create<ReturnOp>(builder.getUnknownLoc(), rootVal);
  module->push_back(spnFunc);
  builder.restoreInsertionPoint(restore);
}

mlir::FuncOp MLIRCodeGen::createSPNFunction(uint32_t numInputs, const std::string& funcName) {
  Type elementType = builder.getIntegerType(32);
  SmallVector<Type, 10> inputs(numInputs, elementType);
  auto func_type = builder.getFunctionType(inputs, {builder.getF64Type()});
  return FuncOp::create(builder.getUnknownLoc(), funcName, func_type);
}
