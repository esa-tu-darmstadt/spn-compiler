//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNtoStandard/Vectorization/BatchVectorizationPatterns.h"
#include "math.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/Support/Debug.h"
#include "SPN/SPNAttributes.h"

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
    op.emitWarning() << "Aborting vectorization: Cannot vectorize Gaussian leaf as the requested input type"
                     << featureType << " cannot be converted to the data-type for computation " << floatResultType
                     << " in vectorized mode";
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
  auto exp = rewriter.create<mlir::math::ExpOp>(op.getLoc(), fraction);
  // e^(-(x - mean)^2/2*variance)) * 1/sqrt(2*PI*variance)
  rewriter.replaceOpWithNewOp<mlir::MulFOp>(op, coefficientConst, exp);
  return success();
}

// Anonymous namespace holding helper functions.
namespace {

  template<typename SourceOp>
  mlir::LogicalResult replaceOpWithGatherFromGlobalMemref(SourceOp op,
                                                          mlir::ConversionPatternRewriter& rewriter,
                                                          mlir::Value indexOperand,
                                                          llvm::ArrayRef<mlir::Attribute> arrayValues) {
    static int tableCount = 0;
    auto resultType = op.getResult().getType();
    auto inputType = indexOperand.getType();
    if (!inputType.template isa<mlir::VectorType>()) {
      // This pattern only handles vectorized implementations and fails if the input is not a vector.
      return mlir::failure();
    }

    // Construct a DenseElementsAttr to hold the array values.
    auto rankedType = mlir::RankedTensorType::get({(long) arrayValues.size()},
                                                  resultType);
    auto valArrayAttr = mlir::DenseElementsAttr::get(rankedType, arrayValues);

    // Set the insertion point to the body of the module (outside the function/kernel).
    auto module = op->template getParentOfType<mlir::ModuleOp>();
    auto restore = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(module.getBody());

    // Construct a global, constant Memref with private visibility, holding the values of the array.
    auto symbolName = "vec_table_" + std::to_string(tableCount++);
    auto visibility = rewriter.getStringAttr("private");
    auto memrefType = mlir::MemRefType::get({(long) arrayValues.size()}, resultType);
    (void) rewriter.create<mlir::GlobalMemrefOp>(op.getLoc(), symbolName, visibility,
                                                              mlir::TypeAttr::get(memrefType), valArrayAttr, true);
    // Restore insertion point
    rewriter.restoreInsertionPoint(restore);

    // Use GetGlobalMemref operation to access the global created above.
    auto addressOf = rewriter.template create<mlir::GetGlobalMemrefOp>(op.getLoc(), memrefType, symbolName);
    auto vectorShape = inputType.template dyn_cast<mlir::VectorType>().getShape();
    // Convert the input to integer type if necessary.
    mlir::Value index = indexOperand;
    auto indexType = inputType.template dyn_cast<mlir::VectorType>().getElementType();
    if (!indexType.isIntOrIndex()) {
      if (indexType.template isa<mlir::FloatType>()) {
        index = rewriter.template create<mlir::FPToUIOp>(op.getLoc(), index,
                                                         mlir::VectorType::get(vectorShape, rewriter.getI64Type()));
      } else {
        // The input type is neither int/index nor float, conversion unknown, fail this pattern.
        return mlir::failure();
      }
    }
    // Construct the constant pass-thru value (values used if the mask is false for an element of the vector).
    auto vectorType = mlir::VectorType::get(vectorShape, resultType);
    auto passThru = broadcastVectorConstant(vectorType, 0.0,
                                            rewriter, op->getLoc());
    // Construct the constant mask.
    auto mask = broadcastVectorConstant(mlir::VectorType::get(vectorShape, rewriter.getI1Type()), true,
                                        rewriter, op->getLoc());
    // Replace the source operation with a gather load from the global memref.
    rewriter.template replaceOpWithNewOp<mlir::vector::GatherOp>(op, vectorType, addressOf,
                                                                 index, mask, passThru);
    return mlir::success();
  }

}

mlir::LogicalResult mlir::spn::BatchVectorizeHistogram::matchAndRewrite(mlir::spn::HistogramOp op,
                                                                        llvm::ArrayRef<mlir::Value> operands,
                                                                        mlir::ConversionPatternRewriter& rewriter) const {
  // Check for single operand, i.e. the index value.
  assert(operands.size() == 1);

  // Collect all mappings from input var value to probability value in a map
  // and compute the minimum lower bound & maximum upper bound.
  llvm::DenseMap<int, double> values;
  int minLB = std::numeric_limits<int>::max();
  int maxUB = std::numeric_limits<int>::min();
  for (auto& b : op.bucketsAttr()) {
    auto bucket = b.cast<Bucket>();
    auto lb = bucket.lb().getInt();
    auto ub = bucket.ub().getInt();
    auto val = bucket.val().getValueAsDouble();
    for (int i = lb; i < ub; ++i) {
      values[i] = val;
    }
    minLB = std::min<int>(minLB, lb);
    maxUB = std::max<int>(maxUB, ub);
  }

  // Currently, we assume that all input vars take no values <0.
  if (minLB < 0) {
    return failure();
  }

  auto resultType = op.getResult().getType();
  if (!resultType.isIntOrFloat()) {
    // Currently only handling Int and Float result types.
    return failure();
  }

  // Flatten the map into an array by filling up empty indices with 0 values.
  SmallVector<Attribute, 256> valArray;
  for (int i = 0; i < maxUB; ++i) {
    double indexVal = NAN;
    if (values.count(i)) {
      indexVal = values[i];
    } else {
      // Fill up with 0 if no value was defined by the histogram.
      indexVal = 0;
    }
    // Construct attribute with constant value. Need to distinguish cases here due to different builder methods.
    if (resultType.isIntOrIndex()) {
      valArray.push_back(rewriter.getIntegerAttr(resultType, (int) indexVal));
    } else {
      valArray.push_back(rewriter.getFloatAttr(resultType, indexVal));
    }

  }
  return replaceOpWithGatherFromGlobalMemref<HistogramOp>(op, rewriter, operands[0], valArray);
}

mlir::LogicalResult mlir::spn::BatchVectorizeCategorical::matchAndRewrite(mlir::spn::CategoricalOp op,
                                                                          llvm::ArrayRef<mlir::Value> operands,
                                                                          mlir::ConversionPatternRewriter& rewriter) const {
  // Check for single operand, i.e., the index value.
  assert(operands.size() == 1);

  return replaceOpWithGatherFromGlobalMemref<CategoricalOp>(op, rewriter, operands[0],
                                                            op.probabilitiesAttr().getValue());
}
