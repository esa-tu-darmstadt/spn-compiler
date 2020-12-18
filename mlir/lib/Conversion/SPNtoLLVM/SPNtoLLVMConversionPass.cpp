//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include "SPNtoLLVM/SPNtoLLVMPatterns.h"
#include "SPNtoLLVM/SPNtoLLVMConversionPass.h"
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"

void mlir::spn::SPNtoLLVMConversionPass::runOnOperation() {

  ConversionTarget target(getContext());

  target.addLegalDialect<LLVM::LLVMDialect>();

  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  target.addIllegalDialect<SPNDialect>();
  target.addIllegalDialect<StandardOpsDialect>();

  LLVMTypeConverter typeConverter(&getContext());

  // Create and populate list of pattern used for conversion.
  OwningRewritePatternList patterns;
  populateVectorToSCFConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);
  populateSPNtoLLVMConversionPatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::spn::createSPNtoLLVMConversionPass() {
  return std::make_unique<SPNtoLLVMConversionPass>();
}