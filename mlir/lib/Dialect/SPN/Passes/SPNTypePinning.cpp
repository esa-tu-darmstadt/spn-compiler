//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Attributes.h"
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
      // TODO Currently just uses F64 type, choose actual best type based on error analysis.
      auto pinnedType = Float64Type::get(context);
      patterns.insert<TypePinConstant>(context, pinnedType);
      patterns.insert<TypePinHistogram, TypePinCategorical, TypePinGaussian>(context, pinnedType);
      patterns.insert<TypePinWeightedSum, TypePinProduct, TypePinSum>(context, pinnedType);
      auto op = getOperation();
      FrozenRewritePatternList frozenPatterns(std::move(patterns));
      if(failed(applyPatternsAndFoldGreedily(op.getBodyRegion(), frozenPatterns))){
        signalPassFailure();
      }
    }

  };

}

std::unique_ptr<OperationPass<ModuleOp>> mlir::spn::createSPNTypePinningPass() {
  return std::make_unique<SPNTypePinning>();
}

