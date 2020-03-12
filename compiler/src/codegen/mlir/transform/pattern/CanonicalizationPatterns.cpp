//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <mlir/IR/DialectHooks.h>
#include "CanonicalizationPatterns.h"

using namespace mlir;
using namespace mlir::spn;

PatternMatchResult ReduceWeightedSumOp::matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const {
  if (op.getNumOperands() == 0) {
    rewriter.eraseOp(op.getOperation());
    return matchSuccess();
  }

  if (op.getNumOperands() == 1) {
    auto addend = op.operands()[0];
    auto weight = op.weights()[0].cast<FloatAttr>().getValueAsDouble();

    auto constant = rewriter.create<ConstantOp>(op.getLoc(), weight);
    SmallVector<Value, 2> ops{addend, constant};
    auto product = rewriter.create<ProductOp>(op.getLoc(), ops);
    rewriter.replaceOp(op, {product});
    return matchSuccess();
  }

  SmallVector<Value, 10> operands;
  auto weights = op.weights().getValue();
  auto AI = op.operands().begin();
  for (auto WI = weights.begin(); WI != weights.end(); ++WI, ++AI) {
    // Check if associated weight is 1.0.
    if (WI->cast<FloatAttr>().getValueAsDouble() == 1.0) {
      operands.push_back(*AI);
    } else {
      return matchFailure();
    }
  }
  // If all weights equal to 1.0, replace with simple sum operation.
  auto newSum = rewriter.create<SumOp>(op.getLoc(), operands);
  rewriter.replaceOp(op, {newSum});
  return matchSuccess();
}

PatternMatchResult ConstantFoldWeightedSumOp::matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const {
  SmallVector<Value, 10> operands;
  SmallVector<double, 10> weights;
  double acc = 0.0;
  size_t index = 0;
  size_t constantOperands = 0;
  for (auto operand : op.operands()) {
    if (auto constantOperand = dyn_cast_or_null<ConstantOp>(operand.getDefiningOp())) {
      auto constantValue = constantOperand.value().convertToDouble();
      auto weightValue = op.weights()[index].cast<FloatAttr>().getValueAsDouble();
      acc += constantValue * weightValue;
      ++constantOperands;
    } else {
      operands.push_back(operand);
      weights.push_back(op.weights()[index].cast<FloatAttr>().getValueAsDouble());
    }
    ++index;
  }
  if (constantOperands <= 1) {
    return matchFailure();
  }
  if (acc > 0.0) {
    operands.push_back(rewriter.create<ConstantOp>(op.getLoc(), acc));
    weights.push_back(1.0);
  }
  auto newSum = rewriter.create<WeightedSumOp>(op.getLoc(), operands, weights);
  rewriter.replaceOp(op, {newSum});
  return matchSuccess();
}
