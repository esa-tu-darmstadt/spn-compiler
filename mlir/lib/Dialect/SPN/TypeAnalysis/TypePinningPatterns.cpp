//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "TypePinningPatterns.h"

mlir::LogicalResult mlir::spn::TypePinConstant::matchAndRewrite(mlir::spn::ConstantOp op,
                                                                mlir::PatternRewriter& rewriter) const {
  if (!op.getResult().getType().isa<ProbabilityType>()) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<ConstantOp>(op, newType, op.valueAttr());
  return success();
}

mlir::LogicalResult mlir::spn::TypePinHistogram::matchAndRewrite(mlir::spn::HistogramOp op,
                                                                 mlir::PatternRewriter& rewriter) const {
  if (!op.getResult().getType().isa<ProbabilityType>()) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<HistogramOp>(op, newType, op.index(), op.bucketsAttr(), op.bucketCountAttr());
  return success();
}

mlir::LogicalResult mlir::spn::TypePinCategorical::matchAndRewrite(mlir::spn::CategoricalOp op,
                                                                   mlir::PatternRewriter& rewriter) const {
  if (!op.getResult().getType().isa<ProbabilityType>()) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<CategoricalOp>(op, newType, op.index(), op.probabilitiesAttr());
  return success();
}

mlir::LogicalResult mlir::spn::TypePinGaussian::matchAndRewrite(mlir::spn::GaussianOp op,
                                                                mlir::PatternRewriter& rewriter) const {
  if (!op.getResult().getType().isa<ProbabilityType>()) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<GaussianOp>(op, newType, op.index(), op.mean(), op.stddev());
  return success();
}

mlir::LogicalResult mlir::spn::TypePinWeightedSum::matchAndRewrite(mlir::spn::WeightedSumOp op,
                                                                   mlir::PatternRewriter& rewriter) const {
  if (!op.getResult().getType().isa<ProbabilityType>()) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<WeightedSumOp>(op, newType, op.operands(), op.weights());
  return success();
}
