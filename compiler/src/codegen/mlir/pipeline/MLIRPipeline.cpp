//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "MLIRPipeline.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>
#include <codegen/mlir/transform/passes/SPNMLIRPasses.h>

using namespace mlir;
using namespace spnc;

MLIRPipeline::MLIRPipeline(spnc::ActionWithOutput<ModuleOp>& _input, std::shared_ptr<MLIRContext> _mlirContext)
    : ActionSingleInput<ModuleOp, ModuleOp>{_input}, mlirContext{std::move(_mlirContext)}, pm{mlirContext.get()} {
  pm.addPass(mlir::spn::createSPNSimplificationPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
}

ModuleOp& MLIRPipeline::execute() {
  if (!cached) {
    auto inputModule = input.execute();
    inputModule.dump();
    module = std::make_unique<ModuleOp>(inputModule.clone());
    auto result = pm.run(*module);
    if (result.value == LogicalResult::Failure) {
      throw std::runtime_error("Running the MLIR pass pipeline failed!");
    }
    cached = true;
  }
  return *module;
}