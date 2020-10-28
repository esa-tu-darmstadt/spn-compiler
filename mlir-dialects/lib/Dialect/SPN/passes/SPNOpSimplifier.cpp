//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "../simplification/SimplificationPatterns.h"
#include "SPNPassDetails.h"
#include "SPN/SPNPasses.h"

using namespace mlir;
using namespace mlir::spn;

namespace {

  ///
  /// MLIR pass simplifying the operations from the SPN dialects
  /// through a series of transformations.
  struct SPNOpSimplifier : public SPNOpSimplifierBase<SPNOpSimplifier> {

    void runOnOperation() override {
      OwningRewritePatternList patterns;
      auto* context = &getContext();
      patterns.insert<SplitWeightedSumOp>(context);
      patterns.insert<BinarizeWeightedSumOp>(context);
      patterns.insert<BinarizeSumOp>(context);
      patterns.insert<BinarizeProductOp>(context);
      Operation* op = getOperation();
      applyPatternsAndFoldGreedily(op->getRegions(), patterns);
    }

  };

}

std::unique_ptr<OperationPass<ModuleOp>> mlir::spn::createSPNOpSimplifierPass() {
  return std::make_unique<SPNOpSimplifier>();
}