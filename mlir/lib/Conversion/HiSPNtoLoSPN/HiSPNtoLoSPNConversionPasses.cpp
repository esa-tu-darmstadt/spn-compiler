//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <HiSPNtoLoSPN/QueryPatterns.h>
#include "HiSPNtoLoSPN/ArithmeticPrecisionAnalysis.h"
#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h"
#include "HiSPNtoLoSPN/NodePatterns.h"
#include "LoSPN/LoSPNDialect.h"
#include "HiSPN/HiSPNOps.h"
#include "HiSPNtoLoSPN/HiSPNTypeConverter.h"

using namespace mlir::spn;

void HiSPNtoLoSPNNodeConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<high::HiSPNDialect>();
  target.addLegalDialect<low::LoSPNDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  target.addIllegalOp<high::ProductNode, high::SumNode, high::HistogramNode,
                      high::CategoricalNode, high::GaussianNode,
                      high::RootNode>();

  // Use type analysis to determine data type for actual computation.
  // The concrete type determined by the analysis replaces the abstract
  // probability type used by the HiSPN dialect.
  auto& arithmeticAnalysis = getAnalysis<mlir::spn::ArithmeticPrecisionAnalysis>();
  HiSPNTypeConverter typeConverter(arithmeticAnalysis.getComputationType());

  OwningRewritePatternList patterns;
  mlir::spn::populateHiSPNtoLoSPNNodePatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
  // Explicitly mark the ArithmeticPrecisionAnalysis as preserved, so the
  // QueryConversionPass can use the information, even though the Graph's
  // nodes have already been converted.
  markAnalysesPreserved<ArithmeticPrecisionAnalysis>();
}

std::unique_ptr<mlir::Pass> mlir::spn::createHiSPNtoLoSPNNodeConversionPass() {
  return std::make_unique<HiSPNtoLoSPNNodeConversionPass>();
}

void HiSPNtoLoSPNQueryConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<low::LoSPNDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  target.addIllegalDialect<high::HiSPNDialect>();

  // Use type analysis to determine data type for actual computation.
  // The concrete type determined by the analysis replaces the abstract
  // probability type used by the HiSPN dialect.
  auto arithmeticAnalysis = getCachedAnalysis<ArithmeticPrecisionAnalysis>();
  assert(arithmeticAnalysis && "The arithmetic analysis needs to be preserved after node conversion");
  HiSPNTypeConverter typeConverter(arithmeticAnalysis->get().getComputationType());

  OwningRewritePatternList patterns;
  mlir::spn::populateHiSPNtoLoSPNQueryPatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyFullConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::spn::createHiSPNtoLoSPNQueryConversionPass() {
  return std::make_unique<HiSPNtoLoSPNQueryConversionPass>();
}
