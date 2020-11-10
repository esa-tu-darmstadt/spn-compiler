//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "CanonicalizationPatterns.h"

using namespace mlir;
using namespace mlir::spn;

LogicalResult ReduceWeightedSumOp::matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const {

  if (op.getNumOperands() == 1) {
    // In case of a single operand, we can replace the weighted sum with a simple
    // product of the operand and the associated weight.
    auto addend = op.operands()[0];
    auto weight = op.weights()[0].cast<FloatAttr>().getValueAsDouble();
    auto constant = rewriter.create<ConstantOp>(op.getLoc(), weight);
    SmallVector<Value, 2> ops{addend, constant};
    rewriter.replaceOpWithNewOp<ProductOp>(op, ops);
    return success();
  }

  SmallVector<Value, 10> operands;
  auto weights = op.weights().getValue();
  auto AI = op.operands().begin();
  for (auto WI = weights.begin(); WI != weights.end(); ++WI, ++AI) {
    // Check if associated weight is 1.0.
    if (WI->cast<FloatAttr>().getValueAsDouble() == 1.0) {
      operands.push_back(*AI);
    } else {
      return failure();
    }
  }
  // If all weights equal to 1.0, replace with simple sum operation.
  auto newSum = rewriter.create<SumOp>(op.getLoc(), operands);
  rewriter.replaceOp(op, {newSum});
  return success();
}

LogicalResult ConstantFoldWeightedSumOp::matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const {
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
    return failure();
  }
  if (acc > 0.0) {
    operands.push_back(rewriter.create<ConstantOp>(op.getLoc(), acc));
    weights.push_back(1.0);
  }
  rewriter.replaceOpWithNewOp<WeightedSumOp>(op, operands, weights);
  return success();
}

LogicalResult ConstantFoldSumOp::matchAndRewrite(SumOp op, PatternRewriter& rewriter) const {
  SmallVector<Value, 10> nonConstantOperands;
  auto foldedConstant = constantFoldOperands<SumOp, std::plus<double>>(op, nonConstantOperands, 0.0);
  if (std::get<0>(foldedConstant) <= 1) {
    // If no or only one constant was found, there's not point in replacing the operation.
    return failure();
  }
  if (std::get<1>(foldedConstant) != 0.0 || nonConstantOperands.empty()) {
    // Constant folding appeared, crate new ConstantOp for the folded constant value.
    nonConstantOperands.push_back(rewriter.create<ConstantOp>(op.getLoc(), std::get<1>(foldedConstant)));
  }
  rewriter.replaceOpWithNewOp<SumOp>(op, nonConstantOperands);
  return success();
}

LogicalResult ConstantFoldProductOp::matchAndRewrite(ProductOp op, PatternRewriter& rewriter) const {
  SmallVector<Value, 10> nonConstantOperands;
  auto foldedConstant = constantFoldOperands<ProductOp, std::multiplies<double>>(op, nonConstantOperands, 1.0);
  if (std::get<1>(foldedConstant) == 0.0) {
    // Special case, one constant zero operand will cause the whole product to become 0.0, replace by constant.
    rewriter.replaceOpWithNewOp<ConstantOp>(op, 0.0);
    return success();
  }
  if (std::get<0>(foldedConstant) <= 1) {
    // If no or only one constant was found, there's not point in replacing the operation.
    return failure();
  }
  if (std::get<1>(foldedConstant) != 1.0 || nonConstantOperands.empty()) {
    // Constant folding appeared, crate new ConstantOp for the folded constant value.
    nonConstantOperands.push_back(rewriter.create<ConstantOp>(op.getLoc(), std::get<1>(foldedConstant)));
  }
  rewriter.replaceOpWithNewOp<ProductOp>(op, nonConstantOperands);
  return success();
}