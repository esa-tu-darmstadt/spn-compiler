//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "MLIRCodeGen.h"
#include "SPN/SPNDialect.h"
#include "SPN/SPNOps.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/StandardTypes.h"
#include "MLIRGraphGen.h"
#include "mlir/IR/Verifier.h"
#include "util/Logging.h"

using namespace mlir::spn;
using namespace spnc;

MLIRCodeGen::MLIRCodeGen(ActionWithOutput<IRGraph>& _input,
                         std::string _kernelName,
                         std::shared_ptr<MLIRContext> _context) : ActionSingleInput<IRGraph, ModuleOp>(_input),
                                                                  kernelName{std::move(_kernelName)},
                                                                  context{std::move(_context)}, builder{context.get()} {
  module = std::make_unique<ModuleOp>(mlir::ModuleOp::create(builder.getUnknownLoc()));
}

ModuleOp& MLIRCodeGen::execute() {
  if (!cached) {
    generateMLIR(input.execute());
    cached = true;
  }
  return *module;
}

void MLIRCodeGen::generateMLIR(IRGraph& graph) {
  builder.setInsertionPointToStart(module->getBody());
  auto numInputsAttr = builder.getUI32IntegerAttr(graph.inputs().size());
  // TODO Retrieve information about input value type from compiler input.
  auto inputType = builder.getIntegerType(32, false);
  auto inputTypeAttr = TypeAttr::get(inputType);
  auto query = builder.create<SingleJointQuery>(builder.getUnknownLoc(), numInputsAttr, inputTypeAttr);
  auto block = builder.createBlock(&query.getRegion());
  for (auto* input : graph.inputs()) {
    node2value[input->id()] = query.getRegion().addArgument(inputType);
  }
  builder.setInsertionPointToEnd(block);
  MLIRGraphGen graphGen{builder, node2value};
  graph.rootNode()->accept(graphGen, nullptr);
  auto resultValue = node2value[graph.rootNode()->id()];
  builder.create<mlir::spn::ReturnOp>(builder.getUnknownLoc(), resultValue);
  module->push_back(query);
  module->dump();
  if (failed(::mlir::verify(module->getOperation()))) {
    SPNC_FATAL_ERROR("Verification of the generated MLIR module failed!");
  }
}

