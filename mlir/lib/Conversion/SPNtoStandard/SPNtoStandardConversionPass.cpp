//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNtoStandard/SPNtoStandardPatterns.h"
#include "SPNtoStandard/SPNtoStandardConversionPass.h"
#include "SPNtoStandard/SPNtoStandardTypeConverter.h"

void mlir::spn::SPNtoStandardConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  SPNtoStandardTypeConverter typeConverter;

  target.addIllegalDialect<SPNDialect>();
  target.addLegalOp<HistogramOp>();

  OwningRewritePatternList patterns;
  mlir::spn::populateSPNtoStandardConversionPatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }

}

std::unique_ptr<mlir::Pass> mlir::spn::createSPNtoStandardConversionPass() {
  return std::make_unique<SPNtoStandardConversionPass>();
}