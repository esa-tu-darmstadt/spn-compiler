//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Attributes.h"
#include "SPN/Analysis/SPNErrorEstimation.h"
#include "SPN/SPNPasses.h"
#include "SPNPassDetails.h"
#include "../TypeAnalysis/TypePinningPatterns.h"

using namespace mlir;
using namespace mlir::spn;

namespace {

  struct SPNTypePinning : public SPNTypePinningBase<SPNTypePinning> {

  protected:
    void runOnOperation() override {
      OwningRewritePatternList patterns;
      auto* context = &getContext();
      // ToDo: Confirm that actual best type based on error analysis is chosen.
      auto analysis = getAnalysis<SPNErrorEstimation>();
      auto pinnedType = analysis.getOptimalType();
      // ToDo: Remove when rewriting of constants is successful
      pinnedType = Float64Type::get(context);
      patterns.insert<TypePinConstant>(context, pinnedType);
      patterns.insert<TypePinHistogram, TypePinCategorical, TypePinGaussian>(context, pinnedType);
      patterns.insert<TypePinWeightedSum, TypePinProduct, TypePinSum>(context, pinnedType);
      auto op = getOperation();
      FrozenRewritePatternList frozenPatterns(std::move(patterns));
      applyPatternsAndFoldGreedily(op.getBodyRegion(), frozenPatterns);
    }

  };

}

std::unique_ptr<OperationPass<ModuleOp>> mlir::spn::createSPNTypePinningPass() {
  return std::make_unique<SPNTypePinning>();
}

