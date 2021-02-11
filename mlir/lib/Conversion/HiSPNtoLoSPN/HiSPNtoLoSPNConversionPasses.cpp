//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <HiSPNtoLoSPN/QueryPatterns.h>
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

  // TODO Use type analysis here.
  HiSPNTypeConverter typeConverter(mlir::Float64Type::get(&getContext()));

  OwningRewritePatternList patterns;
  mlir::spn::populateHiSPNtoLoSPNNodePatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
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

  // TODO Use type analysis here.
  HiSPNTypeConverter typeConverter(mlir::Float64Type::get(&getContext()));

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
