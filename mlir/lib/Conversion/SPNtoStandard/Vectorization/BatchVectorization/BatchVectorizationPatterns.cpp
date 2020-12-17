//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNtoStandard/Vectorization/BatchVectorizationPatterns.h"
#include "llvm/Support/Debug.h"

//
// Anonymous namespace holding a bunch of helper functions.
//
namespace {

  template<typename T>
  mlir::ConstantOp broadcastVectorConstant(mlir::VectorType type, T value,
                                           mlir::ConversionPatternRewriter& rewriter, mlir::Location loc) {
    assert(type.hasStaticShape());
    llvm::SmallVector<T, 8> array;
    for (int i = 0; i < type.getNumElements(); ++i) {
      array.push_back(value);
    }
    auto constAttr = mlir::DenseElementsAttr::get(type, (llvm::ArrayRef<T>) array);
    auto constValue = rewriter.create<mlir::ConstantOp>(loc, constAttr);
    return constValue;
  }

}

mlir::LogicalResult mlir::spn::BatchVectorizeGaussian::matchAndRewrite(mlir::spn::GaussianOp op,
                                                                       llvm::ArrayRef<mlir::Value> operands,
                                                                       mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.size() == 1 && "Expecting only a single operand for Gaussian leaf");

  auto feature = operands.front();

  // Check that the operand is a vector of floats.
  if (!feature.getType().isa<VectorType>() ||
      !feature.getType().dyn_cast<VectorType>().getElementType().isa<FloatType>()) {
    return failure();
  }

  auto vectorType = feature.getType().dyn_cast<VectorType>();
  assert(vectorType);
  auto featureType = vectorType.getElementType().dyn_cast<FloatType>();
  assert(featureType);

  // Get the return type and strip the vector type if necessary.
  Type resultType = op.getResult().getType();
  if (resultType.isa<VectorType>()) {
    resultType = resultType.dyn_cast<VectorType>().getElementType();
  }
  // Check the Gaussian returns a float result.
  if (!resultType.isa<FloatType>()) {
    return failure();
  }
  auto floatResultType = resultType.dyn_cast<FloatType>();
  assert(floatResultType);

  // FPTrunc and FPExt currently do not support vector types.
  // Vectorization of a Gaussian must fail if its involves changing the width of
  // the floating type between input (feature) and result.
  if (featureType.getWidth() != floatResultType.getWidth()) {
    llvm::dbgs() << "WARNING: Cannot vectorize Gaussian due to non-matching floating-point types!\n";
    return failure();
  }

  // Calculate Gaussian distribution using e^(-(x - mean)^2/2*variance))/sqrt(2*PI*variance)
  // Variance from standard deviation.
  double variance = op.stddev().convertToDouble() * op.stddev().convertToDouble();
  // 1/sqrt(2*PI*variance)
  double coefficient = 1.0 / (std::sqrt(2.0 * M_PI * variance));
  auto coefficientConst = broadcastVectorConstant(vectorType, coefficient, rewriter, op.getLoc());
  // -1/(2*variance)
  double denominator = -1.0 / (2.0 * variance);
  auto denominatorConst = broadcastVectorConstant(vectorType, denominator, rewriter, op.getLoc());
  // x - mean
  auto meanConst = broadcastVectorConstant(vectorType, op.mean().convertToDouble(), rewriter, op.getLoc());
  auto subtraction = rewriter.create<mlir::SubFOp>(op.getLoc(), feature, meanConst);
  // (x-mean)^2
  auto numerator = rewriter.create<mlir::MulFOp>(op.getLoc(), subtraction, subtraction);
  // -(x-mean)^2 / 2*variance
  auto fraction = rewriter.create<mlir::MulFOp>(op.getLoc(), numerator, denominatorConst);
  // e^(-(x-mean)^2 / 2*variance)
  auto exp = rewriter.create<mlir::ExpOp>(op.getLoc(), fraction);
  // e^(-(x - mean)^2/2*variance)) * 1/sqrt(2*PI*variance)
  rewriter.replaceOpWithNewOp<mlir::MulFOp>(op, coefficientConst, exp);
  return success();

}