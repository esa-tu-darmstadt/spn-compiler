//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNTransformations.h"
#include "LoSPN/LoSPNPasses.h"
#include "mlir/Transforms/Passes.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/Analysis/SPNBitWidth.h"
#include "toolchain/MLIRToolchain.h"
#include "option/GlobalOptions.h"
#include "util/Logging.h"
#include "pipeline/steps/hdl/EmbedController.hpp"

void spnc::LoSPNTransformations::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  auto maxTaskSize = spnc::option::maxTaskSize.get(*getContext()->get<Configuration>());
  pm->nest<mlir::spn::low::SPNKernel>().addPass(mlir::spn::low::createLoSPNPartitionerPass(maxTaskSize));
  pm->addPass(mlir::spn::low::createLoSPNBufferizePass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->nest<mlir::spn::low::SPNKernel>().addPass(mlir::spn::low::createLoSPNCopyRemovalPass());
  pm->addPass(mlir::createCSEPass());
}

void spnc::LoSPNTransformations::preProcess(mlir::ModuleOp* inputModule) {
  // Pre-processing before bufferization: Find the Kernel with the corresponding
  // name in the module and retrieve information about the data-type and shape
  // of the result values from its function-like type.
  auto kernelInfo = getContext()->get<KernelInfo>();
  for (mlir::spn::low::SPNKernel kernel : inputModule->getOps<mlir::spn::low::SPNKernel>()) {
    if (kernel.getName() == kernelInfo->kernelName) {
      assert(kernel.getFunctionType().getNumResults() == 1);
      auto resultType = kernel.getFunctionType().getResult(0).dyn_cast<mlir::TensorType>();
      assert(resultType);
      kernelInfo->numResults = 1;
      assert(resultType.getElementType().isIntOrFloat());
      unsigned bytesPerResult = resultType.getElementTypeBitWidth() / 8;
      kernelInfo->bytesPerResult = bytesPerResult;
      kernelInfo->dtype = translateType(resultType.getElementType());
    }
  }
}

void spnc::LoSPNTransformations::postProcess(mlir::ModuleOp* transformedModule) {
  PipelineContext *context = getContext();
  KernelInfo *kernelInfo = context->get<KernelInfo>();

  if (kernelInfo->target != KernelTarget::FPGA)
    return;

  ::mlir::spn::SPNBitWidth bitWidth(
    transformedModule->getOperation()
  );

  Kernel kernel{FPGAKernel()};
  context->add<Kernel>(std::move(kernel));
  FPGAKernel& fpgaKernel = context->get<Kernel>()->getFPGAKernel();

  fpgaKernel.spnVarCount = kernelInfo->numFeatures;
  fpgaKernel.spnBitsPerVar =
    bitWidth.getBitsPerVar() == 2 ? 2 : round8(bitWidth.getBitsPerVar());
  // TODO: Make this configurable!
  fpgaKernel.spnResultWidth = 32; // double precision float

  fpgaKernel.mAxisControllerWidth = round8(fpgaKernel.spnResultWidth);
  fpgaKernel.sAxisControllerWidth = round8(fpgaKernel.spnBitsPerVar * fpgaKernel.spnVarCount);

  // TODO: Make this parameterizable
  fpgaKernel.memDataWidth = 32;
  fpgaKernel.memAddrWidth = 32;

  fpgaKernel.liteDataWidth = 32;
  fpgaKernel.liteAddrWidth = 32;

  fpgaKernel.kernelId = 123;
}

std::string spnc::LoSPNTransformations::translateType(mlir::Type type) {
  if (type.isInteger(32)) {
    return "int32";
  }
  if (type.isF64()) {
    return "float64";
  }
  if (type.isF32()) {
    return "float32";
  }
  assert(false && "Unreachable");
}