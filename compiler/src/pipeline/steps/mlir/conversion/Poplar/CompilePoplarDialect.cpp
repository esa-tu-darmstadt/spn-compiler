//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "CompilePoplarDialect.h"
#include "mlir-ipu/Target/Poplar/CodegenToLLVMIRTranslation.h"
#include "mlir-ipu/Target/Poplar/GraphCompiler.h"
#include "mlir-ipu/Target/Poplar/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "option/Options.h"
#include "toolchain/MLIRToolchain.h"

using namespace mlir;
using namespace mlir::ipu::poplar;
using namespace spnc;

CompilePoplarDialect::CompilePoplarDialect(
    StepWithResult<mlir::ModuleOp> &input)
    : StepSingleInput(input) {}

ExecutionResult CompilePoplarDialect::executeStep(mlir::ModuleOp *mlirModule) {
  mlir::MLIRContext *ctx =
      this->getContext()->template get<mlir::MLIRContext>();
  mlir::PassManager pm{ctx};

  PoplarConvertCodeletToFuncPassOptions options;
  switch (option::ipuTarget) {
  case IPUTarget::IPU1:
    options.targetArchitecture = TargetArchitecture::IPU1;
    break;
  case IPUTarget::IPU2:
    options.targetArchitecture = TargetArchitecture::IPU2;
    break;
  case IPUTarget::IPU21:
    options.targetArchitecture = TargetArchitecture::IPU21;
    break;
  case IPUTarget::Model:
    options.targetArchitecture = TargetArchitecture::CPU;
    break;
  }

  pm.addPass(mlir::ipu::poplar::createPoplarConvertCodeletToFuncPass(options));
  pm.addNestedPass<mlir::ipu::poplar::CodegenOp>(
      mlir::ipu::poplar::createPoplarConvertCallingConventionPass());
  pm.addNestedPass<mlir::ipu::poplar::CodegenOp>(
      mlir::ipu::poplar::createPoplarWrapCodeletsPass());
  pm.addNestedPass<mlir::ipu::poplar::CodegenOp>(
      mlir::ipu::poplar::createPoplarConvertCodegenToLLVMPass());

  // Enable IR printing if requested via CLI
  if (spnc::option::dumpIR) {
    pm.enableIRPrinting(
        /* Print before every pass*/ [](mlir::Pass *,
                                        mlir::Operation *) { return false; },
        /* Print after every pass*/
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /* Print module scope*/ true,
        /* Print only after change*/ false);
  }

  auto result = pm.run(*mlirModule);
  if (failed(result)) {
    return spnc::failure("Poplar dialect transformations failed");
  }

  auto graphOps = mlirModule->getOps<GraphOp>();
  if (graphOps.empty()) {
    return failure("Missing graph op");
  }
  GraphOp graphOp = *graphOps.begin();

  auto codegenOps = mlirModule->getOps<CodegenOp>();
  if (codegenOps.empty()) {
    return failure("Missing codegen op");
  }
  CodegenOp codegenOp = *codegenOps.begin();

  const auto *target = getContext()->get<::poplar::Target>();

  FailureOr<::poplar::Executable> maybeExecutable =
      compileGraph(graphOp, codegenOp, *target);

  if (failed(maybeExecutable))
    return failure("Failed to compile graph");

  // unsigned programId,
  //             unsigned query_type, unsigned target, unsigned _batchSize,
  //             unsigned _numFeatures, unsigned _bytesPerFeatures,
  //             unsigned numResults, unsigned bytesPerResult,
  //             const std::string &dataType

  auto *kernelInfo = getContext()->get<KernelInfo>();
  unsigned programId = 0;
  kernel = std::make_unique<IPUKernel>(
      std::move(*maybeExecutable), option::ipuTarget, programId,
      kernelInfo->queryType, kernelInfo->target, kernelInfo->batchSize,
      kernelInfo->numFeatures, kernelInfo->bytesPerFeature,
      kernelInfo->numResults, kernelInfo->bytesPerResult, kernelInfo->dtype);
  return success();
}