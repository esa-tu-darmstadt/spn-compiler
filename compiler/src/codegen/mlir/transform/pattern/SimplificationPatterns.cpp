//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SimplificationPatterns.h"

using namespace mlir;
using namespace mlir::spn;

PatternMatchResult BinarizeWeightedSumOp::matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const {
  if (op.getNumOperands() <= 2) {
    return matchFailure();
  }
  auto pivot = llvm::divideCeil(op.getNumOperands(), 2);
  SmallVector<Value, 10> leftAddends;
  SmallVector<Value, 10> rightAddends;
  int count = 0;
  for (auto a : op.operands()) {
    if (count < pivot) {
      leftAddends.push_back(a);
    } else {
      rightAddends.push_back(a);
    }
    ++count;
  }

  SmallVector<double, 10> leftWeights;
  SmallVector<double, 10> rightWeights;
  count = 0;
  auto weights = op.weights().getValue();
  for (auto w : weights) {
    auto doubleValue = w.cast<FloatAttr>().getValueAsDouble();
    if (count < pivot) {
      leftWeights.push_back(doubleValue);
    } else {
      rightWeights.push_back(doubleValue);
    }
    ++count;
  }
  auto leftSum = rewriter.create<WeightedSumOp>(op.getLoc(), leftAddends, leftWeights);
  auto rightSum = rewriter.create<WeightedSumOp>(op.getLoc(), rightAddends, rightWeights);
  SmallVector<Value, 2> ops{leftSum, rightSum};
  SmallVector<double, 2> newWeights{1.0, 1.0};
  auto newSum = rewriter.create<WeightedSumOp>(op.getLoc(), ops, newWeights);
  rewriter.replaceOp(op, {newSum});
  return matchSuccess();
}

PatternMatchResult SplitWeightedSumOp::matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const {
  SmallVector<Value, 10> addends;
  size_t index = 0;
  for (auto a : op.operands()) {
    auto w = op.weights()[index].cast<FloatAttr>().getValueAsDouble();
    auto constant = rewriter.create<ConstantOp>(op.getLoc(), w);
    SmallVector<Value, 2> multiplicands{a, constant};
    addends.push_back(rewriter.create<ProductOp>(op.getLoc(), multiplicands));
    ++index;
  }
  auto newSum = rewriter.create<SumOp>(op.getLoc(), addends);
  rewriter.replaceOp(op, {newSum});
  return matchSuccess();
}
