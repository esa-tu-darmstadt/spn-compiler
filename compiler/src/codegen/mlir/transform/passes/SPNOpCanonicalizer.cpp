//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "SPNMLIRPasses.h"
#include <codegen/mlir/transform/pattern/CanonicalizationPatterns.h>

using namespace mlir;
using namespace mlir::spn;

namespace {

  ///
  /// MLIR pass canonicalizing the operations from the SPN dialect
  /// through a series of transformations.
  struct SPNOpCanonicalizer : public OperationPass<SPNOpCanonicalizer> {

    void runOnOperation() override {
      OwningRewritePatternList patterns;
      auto* context = &getContext();
      patterns.insert<ReduceWeightedSumOp>(context);
      patterns.insert<ConstantFoldWeightedSumOp>(context);
      patterns.insert<ReduceSumOp>(context);
      patterns.insert<ConstantFoldSumOp>(context);
      patterns.insert<ReduceProductOp>(context);
      patterns.insert<ConstantFoldProductOp>(context);
      Operation* op = getOperation();
      applyPatternsGreedily(op->getRegions(), patterns);
    }

  };

}

std::unique_ptr<Pass> mlir::spn::createSPNCanonicalizationPass() {
  return std::make_unique<SPNOpCanonicalizer>();
}
