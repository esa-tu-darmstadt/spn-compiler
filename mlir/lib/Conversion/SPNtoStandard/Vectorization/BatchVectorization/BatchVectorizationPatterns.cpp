//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "BatchVectorizationPatterns.h"

mlir::LogicalResult mlir::spn::BatchVectorizeConstant::matchAndRewrite(mlir::spn::ConstantOp op,
                                                                       mlir::PatternRewriter& rewriter) const {
  if (op.getResult().getType().isa<VectorType>()) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<ConstantOp>(op, createVectorType(op), op.value());
  return success();
}

mlir::LogicalResult mlir::spn::BatchVectorizeHistogram::matchAndRewrite(mlir::spn::HistogramOp op,
                                                                        mlir::PatternRewriter& rewriter) const {
  if (op.getResult().getType().isa<VectorType>()) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<HistogramOp>(op, createVectorType(op), op.index(), op.buckets(), op.bucketCount());
  return success();
}

mlir::LogicalResult mlir::spn::BatchVectorizeCategorical::matchAndRewrite(mlir::spn::CategoricalOp op,
                                                                          mlir::PatternRewriter& rewriter) const {
  if (op.getResult().getType().isa<VectorType>()) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<CategoricalOp>(op, createVectorType(op), op.index(), op.probabilities());
  return success();
}

mlir::LogicalResult mlir::spn::BatchVectorizeGaussian::matchAndRewrite(mlir::spn::GaussianOp op,
                                                                       mlir::PatternRewriter& rewriter) const {
  if (op.getResult().getType().isa<VectorType>()) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<GaussianOp>(op,
                                          createVectorType(op),
                                          op.index(), op.mean(), op.stddev());
  return success();
}
