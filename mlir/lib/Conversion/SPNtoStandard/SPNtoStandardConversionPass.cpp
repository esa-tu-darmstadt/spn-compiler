//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNtoStandard/SPNtoStandardPatterns.h"
#include "SPNtoStandard/Vectorization/BatchVectorizationPatterns.h"
#include "SPNtoStandard/SPNtoStandardConversionPass.h"
#include "SPNtoStandard/SPNtoStandardTypeConverter.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"

void mlir::spn::SPNtoStandardConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  SPNtoStandardTypeConverter typeConverter;

  // Mark the SPN dialect illegal to trigger conversion of all operations from the dialect.
  target.addIllegalDialect<SPNDialect>();

  OwningRewritePatternList patterns;
  mlir::spn::populateSPNtoStandardConversionPatterns(patterns, &getContext(), typeConverter);
  mlir::spn::populateSPNBatchVectorizePatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyFullConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }

}

std::unique_ptr<mlir::Pass> mlir::spn::createSPNtoStandardConversionPass() {
  return std::make_unique<SPNtoStandardConversionPass>();
}
