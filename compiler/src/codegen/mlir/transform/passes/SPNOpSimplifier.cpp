//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "SPNMLIRPasses.h"
#include <codegen/mlir/transform/pattern/SimplificationPatterns.h>

using namespace mlir;
using namespace mlir::spn;

namespace {

  struct SPNOpSimplifier : public OperationPass<SPNOpSimplifier> {

    void runOnOperation() override {
      OwningRewritePatternList patterns;
      auto* context = &getContext();
      patterns.insert<SplitWeightedSumOp>(context);
      patterns.insert<BinarizeWeightedSumOp>(context);
      patterns.insert<BinarizeSumOp>(context);
      patterns.insert<BinarizeProductOp>(context);
      Operation* op = getOperation();
      applyPatternsGreedily(op->getRegions(), patterns);
    }

  };

}

std::unique_ptr<Pass> mlir::spn::createSPNSimplificationPass() {
  return std::make_unique<SPNOpSimplifier>();
}

static PassRegistration<SPNOpSimplifier> pass("spn-simplify", "simplify SPN-dialect operations");

