//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <util/FileSystem.h>
#include <driver/BaseActions.h>
#include <json/Parser.h>
#include <transform/BinaryTreeTransform.h>
#include <codegen/mlir/MLIRCodeGen.h>
#include "MLIRToolchain.h"

using namespace spnc;

std::unique_ptr<Job<mlir::ModuleOp> > MLIRToolchain::constructJobFromFile(const std::string& inputFile) {
  // Construct file input action.
  auto fileInput = std::make_unique<FileInputAction<FileType::SPN_JSON>>(inputFile);
  return constructJob(std::move(fileInput));
}

std::unique_ptr<Job<mlir::ModuleOp> > MLIRToolchain::constructJobFromString(const std::string& inputString) {
  // Construct string input action.
  auto stringInput = std::make_unique<StringInputAction>(inputString);
  return constructJob(std::move(stringInput));
}

std::unique_ptr<Job<mlir::ModuleOp>> MLIRToolchain::constructJob(std::unique_ptr<ActionWithOutput<std::string>> input) {
  std::unique_ptr<Job<mlir::ModuleOp>> job{new Job<mlir::ModuleOp>()};
  // Construct parser to parse JSON from input.
  auto parser = std::make_unique<Parser>(*input);
  // Invoke MLIR code-generation on parsed tree.
  std::string kernelName = "spn_kernel";

  auto mlirCodeGen = std::make_unique<MLIRCodeGen>(*parser, kernelName, std::make_shared<MLIRContext>());

  job->addAction(std::move(input));
  job->addAction(std::move(parser));
  job->setFinalAction(std::move(mlirCodeGen));
  return std::move(job);
}
