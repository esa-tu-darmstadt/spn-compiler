//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <mlir/IR/BlockAndValueMapping.h>
#include "LoSPNtoCPU/Vectorization/VectorizationPatterns.h"
#include "TargetInformation.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "LoSPN/LoSPNAttributes.h"

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

mlir::LogicalResult mlir::spn::VectorizeBatchRead::matchAndRewrite(mlir::spn::low::SPNBatchRead op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  // Replace the vectorized version of a BatchRead with a Gather load from the input memref.
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized BatchRead");
  }
  assert(operands.size() == 2);
  assert(operands[0].getType().isa<MemRefType>());
  assert(operands[1].getType().isa<IndexType>());
  auto memRef = operands[0].getType().dyn_cast<MemRefType>();
  // Assume that the second dimension (i.e., the number of features per sample is a static dimension).
  assert(!memRef.isDynamicDim(1));
  auto numFeatures = memRef.getDimSize(1);
  auto vectorType = VectorType::get({op.vectorFactor()}, op.getResult().getType());
  // Broadcast the batchIndex
  auto vectorOfIndex = VectorType::get(op.vectorFactor(), rewriter.getI64Type());
  auto convertedBatchIndex = rewriter.create<IndexCastOp>(op.getLoc(), rewriter.getI64Type(), operands[1]);
  auto batchIndex = rewriter.create<vector::BroadcastOp>(op.getLoc(), vectorOfIndex, convertedBatchIndex);
  // Create a constant vector with the offsets of the elements from the first sample.
  SmallVector<unsigned long, 4> offsets;
  for (unsigned i = 0; i < op.vectorFactor(); ++i) {
    offsets.push_back(i * numFeatures + op.sampleIndex());
  }
  auto constAttr = mlir::DenseElementsAttr::get(vectorOfIndex, (llvm::ArrayRef<unsigned long>) offsets);
  auto constOffset = rewriter.create<ConstantOp>(op.getLoc(), constAttr);
  // Add the offsets to the base index from the batchIndex.
  auto addresses = rewriter.create<AddIOp>(op.getLoc(), batchIndex, constOffset);
  // Create constant passThru.
  auto passThru = broadcastVectorConstant(vectorType, 0.0,
                                          rewriter, op->getLoc());
  // Construct the constant mask.
  auto mask = broadcastVectorConstant(mlir::VectorType::get(op.vectorFactor(), rewriter.getI1Type()), true,
                                      rewriter, op->getLoc());
  rewriter.replaceOpWithNewOp<vector::GatherOp>(op, vectorType, operands[0], addresses, mask, passThru);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeBatchWrite::matchAndRewrite(mlir::spn::low::SPNBatchWrite op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized BatchWrite");
  }
  assert(operands.size() == 3);
  VectorType vectorType;
  auto result = operands[0];
  if (!result.getType().isa<VectorType>()) {
    vectorType = VectorType::get({op.vectorFactor()}, result.getType());
    result = typeConverter->materializeTargetConversion(rewriter, op->getLoc(), vectorType, result);
    assert(result);
  } else {
    vectorType = result.getType().dyn_cast<VectorType>();
  }
  assert(operands[1].getType().dyn_cast<MemRefType>().getElementType() == vectorType.getElementType()
             && "Result type and element type of MemRef must match");
  assert(operands[1].getType().isa<MemRefType>());
  assert(operands[2].getType().isa<IndexType>());
  rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(op, result, operands[1],
                                                       operands[2]);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeMul::matchAndRewrite(mlir::spn::low::SPNMul op,
                                                             llvm::ArrayRef<mlir::Value> operands,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized Multiplication");
  }
  assert(operands.size() == 2);
  assert(operands[0].getType().isa<VectorType>());
  assert(operands[1].getType().isa<VectorType>());
  rewriter.replaceOpWithNewOp<MulFOp>(op, typeConverter->convertType(op.getResult().getType()),
                                      operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeAdd::matchAndRewrite(mlir::spn::low::SPNAdd op,
                                                             llvm::ArrayRef<mlir::Value> operands,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized Addition");
  }
  assert(operands.size() == 2);
  assert(operands[0].getType().isa<VectorType>());
  assert(operands[1].getType().isa<VectorType>());
  rewriter.replaceOpWithNewOp<AddFOp>(op, typeConverter->convertType(op.getResult().getType()),
                                      operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeLog::matchAndRewrite(low::SPNLog op,
                                                             ArrayRef<Value> operands,
                                                             ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized Logarithm");
  }
  assert(operands.size() == 1);
  assert(operands[0].getType().isa<VectorType>());
  rewriter.replaceOpWithNewOp<math::LogOp>(op, typeConverter->convertType(op.getResult().getType()),
                                           operands[0]);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeConstant::matchAndRewrite(mlir::spn::low::SPNConstant op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized Constant");
  }
  assert(operands.empty());
  assert(op.getResult().getType().isa<FloatType>());
  auto vectorConstantTy = VectorType::get(op.vectorFactor(), op.getResult().getType());
  auto vectorConstant = broadcastVectorConstant(vectorConstantTy, op.value().convertToDouble(),
                                                rewriter, op.getLoc());
  rewriter.replaceOp(op, ValueRange{vectorConstant});
  return success();
}

mlir::LogicalResult mlir::spn::ResolveConvertToVector::matchAndRewrite(mlir::spn::low::SPNConvertToVector op,
                                                                       llvm::ArrayRef<mlir::Value> operands,
                                                                       mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.size() == 1);
  if (operands[0].getType() != op.getResult().getType()) {
    return rewriter.notifyMatchFailure(op, "Conversion to vector cannot be resolved trivially");
  }
  // This handles the case the ConvertToVector was inserted as a materialization, but the input value
  // has been vectorized in the meantime, so the conversion can be trivially resolved.
  rewriter.replaceOp(op, operands);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeGaussian::matchAndRewrite(mlir::spn::low::SPNGaussianLeaf op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized GaussianLeaf");
  }

  assert(operands.size() == 1 && "Expecting only a single operand for Gaussian leaf");
  Value feature = operands.front();

  // Check that the operand is a vector of floats.
  // TODO Handle integer vectors via conversion in vectorized mode.
  if (!feature.getType().isa<VectorType>()) {
    return rewriter.notifyMatchFailure(op, "Vectorization pattern did not match, input was not a vector");
  }
  auto vectorType = feature.getType().dyn_cast<VectorType>();
  assert(vectorType);
  // Get the return type
  Type resultType = op.getResult().getType();
  // Check the Gaussian returns a float result.
  if (!resultType.isa<FloatType>()) {
    return failure();
  }
  auto floatResultType = resultType.dyn_cast<FloatType>();
  assert(floatResultType);
  // Convert from integer input to floating-point value if necessary.
  // This conversion is also possible in vectorized mode.
  if (vectorType.getElementType().isIntOrIndex()) {
    auto floatVectorTy = VectorType::get(vectorType.getShape(), floatResultType);
    feature = rewriter.create<UIToFPOp>(op->getLoc(), feature, floatVectorTy);
  }
  auto featureType = vectorType.getElementType().dyn_cast<FloatType>();
  assert(featureType);

  // FPTrunc and FPExt currently do not support vector types.
  // Vectorization of a Gaussian must fail if its involves changing the width of
  // the floating type between input (feature) and result.
  if (featureType.getWidth() != floatResultType.getWidth()) {
    return rewriter.notifyMatchFailure(op,
                                       StringRef("Aborting vectorization: ") +
                                           "Cannot vectorize Gaussian leaf as the requested input type" +
                                           llvm::formatv("{}", featureType) +
                                           " cannot be converted to the data-type for computation" +
                                           llvm::formatv("{}", floatResultType) +
                                           " in vectorized mode");
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
    auto globalMemref = rewriter.create<mlir::GlobalMemrefOp>(op.getLoc(), symbolName, visibility,
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

mlir::LogicalResult mlir::spn::VectorizeCategorical::matchAndRewrite(mlir::spn::low::SPNCategoricalLeaf op,
                                                                     llvm::ArrayRef<mlir::Value> operands,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized CategoricalLeaf");
  }
  // Check for single operand, i.e., the index value.
  assert(operands.size() == 1);

  return replaceOpWithGatherFromGlobalMemref<low::SPNCategoricalLeaf>(op, rewriter, operands[0],
                                                                      op.probabilitiesAttr().getValue());
}

mlir::LogicalResult mlir::spn::VectorizeHistogram::matchAndRewrite(mlir::spn::low::SPNHistogramLeaf op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized HistogramLeaf");
  }
  // Check for single operand, i.e. the index value.
  assert(operands.size() == 1);

  // Collect all mappings from input var value to probability value in a map
  // and compute the minimum lower bound & maximum upper bound.
  llvm::DenseMap<int, double> values;
  int minLB = std::numeric_limits<int>::max();
  int maxUB = std::numeric_limits<int>::min();
  for (auto& b : op.bucketsAttr()) {
    auto bucket = b.cast<low::Bucket>();
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
    double indexVal;
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
  return replaceOpWithGatherFromGlobalMemref<low::SPNHistogramLeaf>(op, rewriter, operands[0], valArray);
}
