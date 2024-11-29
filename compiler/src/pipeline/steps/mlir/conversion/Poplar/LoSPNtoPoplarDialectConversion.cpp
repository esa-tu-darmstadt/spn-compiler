//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoPoplarDialectConversion.h"
#include "LoSPNtoPoplar/LoSPNtoPoplarConversionPasses.h"
#include "mlir-ipu/Target/Poplar/Passes.h"

using namespace mlir;

void spnc::LoSPNtoPoplarDialectConversion::initializePassPipeline(
    mlir::PassManager *pm, mlir::MLIRContext *ctx) {
  pm->addPass(mlir::spn::low::createLoSPNtoPoplarConversionPass());

  //   pm->addPass(mlir::ipu::poplar::createPoplarConvertCodeletToFuncPass());
  //   pm->addNestedPass<mlir::ipu::poplar::CodegenOp>(
  //       mlir::ipu::poplar::createPoplarConvertCallingConventionPass());
  //   pm->addNestedPass<mlir::ipu::poplar::CodegenOp>(
  //       mlir::ipu::poplar::createPoplarWrapCodeletsPass());
  //   pm->addNestedPass<mlir::ipu::poplar::CodegenOp>(
  //       mlir::ipu::poplar::createPoplarConvertCodegenToLLVMPass());
}