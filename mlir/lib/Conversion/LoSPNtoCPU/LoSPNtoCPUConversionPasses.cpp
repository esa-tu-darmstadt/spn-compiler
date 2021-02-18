//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/LoSPNtoCPUConversionPasses.h"
#include "LoSPNtoCPU/LoSPNtoCPUTypeConverter.h"
#include "LoSPNtoCPU/StructurePatterns.h"
#include "LoSPNtoCPU/NodePatterns.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

void mlir::spn::LoSPNtoCPUStructureConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  LoSPNtoCPUTypeConverter typeConverter;

  target.addLegalDialect<mlir::spn::low::LoSPNDialect>();
  target.addIllegalOp<mlir::spn::low::SPNKernel>();
  target.addIllegalOp<mlir::spn::low::SPNTask, mlir::spn::low::SPNBody>();

  OwningRewritePatternList patterns;
  mlir::spn::populateLoSPNtoCPUStructurePatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }

}

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNtoCPUStructureConversionPass() {
  return std::make_unique<LoSPNtoCPUStructureConversionPass>();
}

void mlir::spn::LoSPNtoCPUNodeConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::math::MathDialect>();
  // Linalg is required here, because we lower spn.copy to linalg.copy
  // as the Standard dialect currently does not have a copy operation.
  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  LoSPNtoCPUTypeConverter typeConverter;

  target.addIllegalDialect<mlir::spn::low::LoSPNDialect>();

  OwningRewritePatternList patterns;
  mlir::spn::populateLoSPNtoCPUNodePatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyFullConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNtoCPUNodeConversionPass() {
  return std::make_unique<LoSPNtoCPUNodeConversionPass>();
}
