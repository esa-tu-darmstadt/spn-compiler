//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoGPU/LoSPNtoGPUConversionPasses.h"
#include "LoSPNtoGPU/LoSPNtoGPUTypeConverter.h"
#include "LoSPNtoGPU/GPUStructurePatterns.h"
#include "LoSPNtoGPU/GPUNodePatterns.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"

void mlir::spn::LoSPNtoGPUStructureConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::gpu::GPUDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  LoSPNtoGPUTypeConverter typeConverter;

  target.addLegalDialect<mlir::spn::low::LoSPNDialect>();
  target.addIllegalOp<mlir::spn::low::SPNKernel>();
  target.addIllegalOp<mlir::spn::low::SPNTask, mlir::spn::low::SPNBody>();

  OwningRewritePatternList patterns;
  mlir::spn::populateLoSPNtoGPUStructurePatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNtoGPUStructureConversionPass() {
  return std::make_unique<LoSPNtoGPUStructureConversionPass>();
}

void mlir::spn::LoSPNtoGPUNodeConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::math::MathDialect>();
  // Linalg is required here, because we lower spn.copy to linalg.copy
  // as the Standard dialect currently does not have a copy operation.
  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  target.addLegalDialect<mlir::gpu::GPUDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  LoSPNtoGPUTypeConverter typeConverter;

  target.addIllegalDialect<mlir::spn::low::LoSPNDialect>();

  OwningRewritePatternList patterns;
  mlir::spn::populateLoSPNtoGPUNodePatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyFullConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNtoGPUNodeConversionPass() {
  return std::make_unique<LoSPNtoGPUNodeConversionPass>();
}
