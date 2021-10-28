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
#include "toolchain/MLIRToolchain.h"
#include "option/GlobalOptions.h"
#include "util/Logging.h"

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
      assert(kernel.getType().getNumResults() == 1);
      auto resultType = kernel.getType().getResult(0).dyn_cast<mlir::TensorType>();
      assert(resultType);
      kernelInfo->numResults = 1;
      assert(resultType.getElementType().isIntOrFloat());
      unsigned bytesPerResult = resultType.getElementTypeBitWidth() / 8;
      kernelInfo->bytesPerResult = bytesPerResult;
      kernelInfo->dtype = translateType(resultType.getElementType());
    }
  }
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