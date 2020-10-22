//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <util/FileSystem.h>
#include <driver/BaseActions.h>
#include <driver/GlobalOptions.h>
#include <frontend/json/Parser.h>
#include <codegen/mlir/codegen/MLIRCodeGen.h>
#include <SPN/SPNDialect.h>
#include "MLIRToolchain.h"
#include "mlir/InitAllDialects.h"

using namespace spnc;

std::unique_ptr<Job<ModuleOp> > MLIRToolchain::constructJobFromFile(const std::string& inputFile,
                                                                    const Configuration& config) {
  // Construct file input action.
  auto fileInput = std::make_unique<FileInputAction>(inputFile);
  return constructJob(std::move(fileInput), config);
}

std::unique_ptr<Job<ModuleOp> > MLIRToolchain::constructJobFromString(const std::string& inputString,
                                                                      const Configuration& config) {
  // Construct string input action.
  auto stringInput = std::make_unique<StringInputAction>(inputString);
  return constructJob(std::move(stringInput), config);
}

std::unique_ptr<Job<ModuleOp>> MLIRToolchain::constructJob(std::unique_ptr<ActionWithOutput<std::string>> input,
                                                           const Configuration& config) {
  std::unique_ptr<Job<ModuleOp>> job{new Job<ModuleOp>()};
  // Construct parser to parse JSON from input.
  auto graphIRContext = std::make_shared<GraphIRContext>();
  auto& parser = job->insertAction<Parser>(*input, graphIRContext);
  // Invoke MLIR code-generation on parsed tree.
  std::string kernelName = "spn_kernel";
  mlir::registerAllDialects();
  mlir::registerDialect<mlir::spn::SPNDialect>();
  // Register our Dialect with MLIR.
  auto ctx = std::make_shared<MLIRContext>(true);
  ctx->loadDialect<mlir::spn::SPNDialect>();
  for (auto d : ctx->getAvailableDialects()) {
    std::cout << d.str() << std::endl;
  }
  auto& mlirCodeGen = job->insertFinalAction<MLIRCodeGen>(parser, kernelName, ctx);

  job->addAction(std::move(input));
  return std::move(job);
}
