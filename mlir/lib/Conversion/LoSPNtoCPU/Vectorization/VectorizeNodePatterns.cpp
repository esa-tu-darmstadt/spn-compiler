//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <mlir/IR/BlockAndValueMapping.h>
#include "LoSPNtoCPU/Vectorization/VectorizationPatterns.h"
#include "../Target/TargetInformation.h"
#include "llvm/Support/FormatVariadic.h"
#include <cmath>
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

  template<>
  mlir::ConstantOp broadcastVectorConstant(mlir::VectorType type, double value,
                                           mlir::ConversionPatternRewriter& rewriter, mlir::Location loc) {
    assert(type.hasStaticShape());
    assert(type.getElementType().isa<mlir::FloatType>());
    auto floatType = type.getElementType().cast<mlir::FloatType>();
    assert(floatType.getWidth() == 32 || floatType.getWidth() == 64);
    if (floatType.getWidth() == 32) {
      llvm::SmallVector<float, 8> array;
      for (int i = 0; i < type.getNumElements(); ++i) {
        array.push_back((float) value);
      }
      auto constAttr = mlir::DenseElementsAttr::get(type, (llvm::ArrayRef<float>) array);
      auto constValue = rewriter.create<mlir::ConstantOp>(loc, constAttr);
      return constValue;
    } else {
      llvm::SmallVector<double, 8> array;
      for (int i = 0; i < type.getNumElements(); ++i) {
        array.push_back(value);
      }
      auto constAttr = mlir::DenseElementsAttr::get(type, (llvm::ArrayRef<double>) array);
      auto constValue = rewriter.create<mlir::ConstantOp>(loc, constAttr);
      return constValue;
    }
  }

  mlir::Value extendTruncateOrGetVector(mlir::Value input,
                                        mlir::VectorType targetVectorType,
                                        mlir::ConversionPatternRewriter& rewriter) {
    auto inputVectorType = input.getType().dyn_cast<mlir::VectorType>();
    assert(inputVectorType && "cannot extend or truncate scalar type to vector type");
    assert(targetVectorType.getElementType().isa<mlir::FloatType>() && "target element type must be float type");
    if (inputVectorType.getElementTypeBitWidth() < targetVectorType.getElementTypeBitWidth()) {
      return rewriter.create<mlir::FPExtOp>(input.getLoc(), input, targetVectorType);
    } else if (inputVectorType.getElementTypeBitWidth() > targetVectorType.getElementTypeBitWidth()) {
      return rewriter.create<mlir::FPTruncOp>(input.getLoc(), input, targetVectorType);
    }
    return input;
  }

}

mlir::LogicalResult mlir::spn::VectorizeTransposedBatchRead::matchAndRewrite(mlir::spn::low::SPNBatchRead op,
                                                                             llvm::ArrayRef<mlir::Value> operands,
                                                                             mlir::ConversionPatternRewriter& rewriter) const {
  // Replace the vectorized version of a transposed BatchRead with a vector load from the input memref.
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized BatchRead");
  }
  if (!op.transposed().getValueOr(false)) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches transposed BatchRead");
  }
  assert(operands.size() == 2);
  assert(operands[0].getType().isa<MemRefType>());
  assert(operands[1].getType().isa<IndexType>());
  auto memRef = operands[0].getType().dyn_cast<MemRefType>();
  assert(memRef.hasRank() && memRef.getRank() == 2);
  SmallVector<Value, 2> indices;
  auto constStaticIndex = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(op.staticIndex()));
  indices.push_back(constStaticIndex);
  indices.push_back(operands[1]);
  auto vectorType = VectorType::get({op.vectorFactor()}, memRef.getElementType());
  rewriter.replaceOpWithNewOp<vector::TransferReadOp>(op, vectorType, operands[0], indices);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeBatchRead::matchAndRewrite(mlir::spn::low::SPNBatchRead op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  // Replace the vectorized version of a non-transposed BatchRead with a Gather load from the input memref.
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized BatchRead");
  }
  if (op.transposed().getValueOr(false)) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches non-transposed BatchRead");
  }
  assert(operands.size() == 2);
  assert(operands[0].getType().isa<MemRefType>());
  assert(operands[1].getType().isa<IndexType>());
  auto memRef = operands[0].getType().dyn_cast<MemRefType>();
  assert(memRef.hasRank() && memRef.getRank() == 2);
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
    offsets.push_back(i * numFeatures + op.staticIndex());
  }
  auto constAttr = mlir::DenseElementsAttr::get(vectorOfIndex, (llvm::ArrayRef<unsigned long>) offsets);
  auto constOffset = rewriter.create<ConstantOp>(op.getLoc(), constAttr);
  // Multiply the batchIndex with the number of features for the base address.
  auto elements = broadcastVectorConstant(batchIndex.getVectorType(), numFeatures, rewriter, op->getLoc());
  auto baseAddress = rewriter.create<MulIOp>(op->getLoc(), batchIndex, elements);
  // Add the offsets to the base index from the batchIndex.
  auto addresses = rewriter.create<AddIOp>(op.getLoc(), baseAddress, constOffset);
  // Create constant passThru.
  assert(vectorType.getElementType().isIntOrIndexOrFloat());
  Value passThru;
  if (vectorType.getElementType().isIntOrIndex()) {
    passThru = broadcastVectorConstant(vectorType, 0, rewriter, op->getLoc());
  } else if (vectorType.getElementType().isa<FloatType>()) {
    passThru = broadcastVectorConstant(vectorType, 0.0, rewriter, op->getLoc());
  }
  // Construct the constant mask.
  auto mask = broadcastVectorConstant(mlir::VectorType::get(op.vectorFactor(), rewriter.getI1Type()), true,
                                      rewriter, op->getLoc());
  // Re-interpret the MemRef to a single dimension for use with the gather-instruction.
  auto numSamples = rewriter.create<memref::DimOp>(op.getLoc(), operands[0], 0);
  auto constNumFeatures = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(numFeatures));
  auto size = rewriter.create<MulIOp>(op->getLoc(), numSamples, constNumFeatures);
  auto staticOffset = rewriter.getI32IntegerAttr(0);
  auto staticStride = rewriter.getI32IntegerAttr(1);
  SmallVector<OpFoldResult, 1> dynamicSizes;
  dynamicSizes.push_back(size.result());
  SmallVector<OpFoldResult, 1> staticStrides;
  staticStrides.push_back(staticStride);
  auto reinterpret = rewriter.create<memref::ReinterpretCastOp>(op.getLoc(),
                                                                MemRefType::get({-1}, memRef.getElementType()),
                                                                operands[0], staticOffset,
                                                                dynamicSizes, staticStrides);
  Value constIndex = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  rewriter.replaceOpWithNewOp<vector::GatherOp>(op, vectorType, reinterpret, constIndex, addresses, mask, passThru);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeBatchWrite::matchAndRewrite(mlir::spn::low::SPNBatchWrite op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized BatchWrite");
  }
  if (!op.transposed().getValueOr(false)) {
    // Currently, no step of the compilation pipeline will create non-transposed BatchWrite,
    // therefore this is currently the only implementation.
    return rewriter.notifyMatchFailure(op, "Pattern only matches transposed BatchWrite");
  }
  auto memRef = operands[0];
  auto memRefTy = memRef.getType().dyn_cast<MemRefType>();
  assert(memRefTy);
  assert(memRefTy.hasRank() && memRefTy.getRank() == 2);
  auto dynIndex = operands[1];
  assert(dynIndex.getType().isa<IndexType>());

  for (unsigned i = 0; i < op.resultValues().size(); ++i) {
    VectorType vectorType;
    auto result = operands[i + 2];
    if (!result.getType().isa<VectorType>()) {
      vectorType = VectorType::get({op.vectorFactor()}, result.getType());
      result = typeConverter->materializeTargetConversion(rewriter, op->getLoc(), vectorType, result);
      assert(result);
    } else {
      vectorType = result.getType().cast<VectorType>();
    }
    assert(memRefTy.getElementType() == vectorType.getElementType()
               && "Result type and element type of MemRef must match");
    SmallVector<Value, 2> indices;
    auto constStaticIndex = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i));
    indices.push_back(constStaticIndex);
    indices.push_back(dynIndex);
    rewriter.create<vector::TransferWriteOp>(op.getLoc(), result, memRef,
                                             indices);
  }
  rewriter.eraseOp(op);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeMul::matchAndRewrite(mlir::spn::low::SPNMul op,
                                                             llvm::ArrayRef<mlir::Value> operands,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized Multiplication");
  }
  if (op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not match for log-space computation");
  }
  assert(operands.size() == 2);
  assert(operands[0].getType().isa<VectorType>());
  assert(operands[1].getType().isa<VectorType>());
  rewriter.replaceOpWithNewOp<MulFOp>(op, typeConverter->convertType(op.getResult().getType()),
                                      operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeLogMul::matchAndRewrite(mlir::spn::low::SPNMul op,
                                                                llvm::ArrayRef<mlir::Value> operands,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized Multiplication");
  }
  if (!op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches for log-space computation");
  }
  assert(operands.size() == 2);
  assert(operands[0].getType().isa<VectorType>());
  assert(operands[1].getType().isa<VectorType>());
  rewriter.replaceOpWithNewOp<AddFOp>(op, typeConverter->convertType(op.getResult().getType()),
                                      operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeAdd::matchAndRewrite(mlir::spn::low::SPNAdd op,
                                                             llvm::ArrayRef<mlir::Value> operands,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized Addition");
  }
  if (op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not match for log-space computation");
  }
  assert(operands.size() == 2);
  assert(operands[0].getType().isa<VectorType>());
  assert(operands[1].getType().isa<VectorType>());
  rewriter.replaceOpWithNewOp<AddFOp>(op, typeConverter->convertType(op.getResult().getType()),
                                      operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeLogAdd::matchAndRewrite(mlir::spn::low::SPNAdd op,
                                                                llvm::ArrayRef<mlir::Value> operands,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized Addition");
  }
  if (!op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches for log-space computation");
  }
  assert(operands.size() == 2);
  assert(operands[0].getType().isa<VectorType>());
  assert(operands[1].getType().isa<VectorType>());
  // Calculate addition 'x + y' in log-space as
  // 'a + log(1 + exp(b-a)', with a == log(x),
  // b == log(y) and a > b.
  auto compare = rewriter.create<CmpFOp>(op.getLoc(), CmpFPredicate::OGT, operands[0], operands[1]);
  auto a = rewriter.create<SelectOp>(op->getLoc(), compare, operands[0], operands[1]);
  auto b = rewriter.create<SelectOp>(op->getLoc(), compare, operands[1], operands[0]);
  auto sub = rewriter.create<SubFOp>(op->getLoc(), b, a);
  auto exp = rewriter.create<math::ExpOp>(op.getLoc(), sub);
  auto log = rewriter.create<math::Log1pOp>(op.getLoc(), exp);
  rewriter.replaceOpWithNewOp<AddFOp>(op, a, log);
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
  Type resultType = op.getResult().getType();
  if (auto logType = resultType.dyn_cast<low::LogType>()) {
    resultType = logType.getBaseType();
  }
  assert(resultType.isa<FloatType>());
  auto vectorConstantTy = VectorType::get(op.vectorFactor(), resultType);
  auto vectorConstant = broadcastVectorConstant(vectorConstantTy, op.value().convertToDouble(),
                                                rewriter, op.getLoc());
  rewriter.replaceOp(op, ValueRange{vectorConstant});
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeGaussian::matchAndRewrite(mlir::spn::low::SPNGaussianLeaf op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized GaussianLeaf");
  }
  if (op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not match for log-space computation");
  }

  assert(operands.size() == 1 && "Expecting only a single operand for Gaussian leaf");
  Value feature = operands.front();

  if (!feature.getType().isa<VectorType>()) {
    return rewriter.notifyMatchFailure(op, "Vectorization pattern did not match, input was not a vector");
  }
  VectorType vectorType = feature.getType().dyn_cast<VectorType>();
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
    vectorType = floatVectorTy;
  }
  auto featureType = vectorType.getElementType().dyn_cast<FloatType>();
  assert(featureType);

  auto targetVectorType = mlir::VectorType::get(vectorType.getShape(), floatResultType);
  feature = extendTruncateOrGetVector(feature, targetVectorType, rewriter);

  // Calculate Gaussian distribution using e^(-(x - mean)^2/2*variance))/sqrt(2*PI*variance)
  // Variance from standard deviation.
  double variance = op.stddev().convertToDouble() * op.stddev().convertToDouble();
  // 1/sqrt(2*PI*variance)
  double coefficient = 1.0 / (std::sqrt(2.0 * M_PI * variance));
  auto coefficientConst = broadcastVectorConstant(targetVectorType, coefficient, rewriter, op.getLoc());
  // -1/(2*variance)
  double denominator = -1.0 / (2.0 * variance);
  auto denominatorConst = broadcastVectorConstant(targetVectorType, denominator, rewriter, op.getLoc());
  // x - mean
  auto meanConst = broadcastVectorConstant(targetVectorType, op.mean().convertToDouble(), rewriter, op.getLoc());
  auto subtraction = rewriter.create<mlir::SubFOp>(op.getLoc(), feature, meanConst);
  // (x-mean)^2
  auto numerator = rewriter.create<mlir::MulFOp>(op.getLoc(), subtraction, subtraction);
  // -(x-mean)^2 / 2*variance
  auto fraction = rewriter.create<mlir::MulFOp>(op.getLoc(), numerator, denominatorConst);
  // e^(-(x-mean)^2 / 2*variance)
  auto exp = rewriter.create<mlir::math::ExpOp>(op.getLoc(), fraction);
  // e^(-(x - mean)^2/2*variance)) * 1/sqrt(2*PI*variance)
  Value gaussian = rewriter.create<mlir::MulFOp>(op->getLoc(), coefficientConst, exp);
  if (op.supportMarginal()) {
    auto isNan = rewriter.create<mlir::CmpFOp>(op->getLoc(), CmpFPredicate::UNO, feature, feature);
    auto constOne = broadcastVectorConstant(targetVectorType, 1.0, rewriter, op.getLoc());
    gaussian = rewriter.create<mlir::SelectOp>(op.getLoc(), isNan, constOne, gaussian);
  }
  rewriter.replaceOp(op, gaussian);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeLogGaussian::matchAndRewrite(mlir::spn::low::SPNGaussianLeaf op,
                                                                     llvm::ArrayRef<mlir::Value> operands,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized GaussianLeaf");
  }
  if (!op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches for log-space computation");
  }

  assert(operands.size() == 1 && "Expecting only a single operand for Gaussian leaf");
  Value feature = operands.front();

  if (!feature.getType().isa<VectorType>()) {
    return rewriter.notifyMatchFailure(op, "Vectorization pattern did not match, input was not a vector");
  }
  VectorType vectorType = feature.getType().dyn_cast<VectorType>();
  assert(vectorType);
  // Get the return type
  Type resultType = op.getResult().getType().cast<low::LogType>().getBaseType();
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
    vectorType = floatVectorTy;
  }
  auto featureType = vectorType.getElementType().dyn_cast<FloatType>();
  assert(featureType);

  auto targetVectorType = mlir::VectorType::get(vectorType.getShape(), floatResultType);
  feature = extendTruncateOrGetVector(feature, targetVectorType, rewriter);

  // Calculate Gaussian distribution using the logarithm of the PDF of the Normal (Gaussian) distribution,
  // given as '-ln(stddev) - 1/2 ln(2*pi) - (x - mean)^2 / 2*stddev^2'
  // First term, -ln(stddev)
  double firstTerm = -log(op.stddev().convertToDouble());
  // Second term, - 1/2 ln(2*pi)
  double secondTerm = -0.5 * log(2 * M_PI);
  // Denominator, - 1/2*(stddev^2)
  double denominator = -(1.0 / (2.0 * op.stddev().convertToDouble() * op.stddev().convertToDouble()));
  auto denominatorConst = broadcastVectorConstant(targetVectorType, denominator, rewriter, op->getLoc());
  // Coefficient, summing up the first two constant terms
  double coefficient = firstTerm + secondTerm;
  auto coefficientConst = broadcastVectorConstant(targetVectorType, coefficient, rewriter, op.getLoc());
  // x - mean
  auto meanConst = broadcastVectorConstant(targetVectorType, op.meanAttr().getValueAsDouble(), rewriter, op.getLoc());

  auto subtraction = rewriter.create<mlir::SubFOp>(op.getLoc(), feature, meanConst);
  // (x-mean)^2
  auto numerator = rewriter.create<mlir::MulFOp>(op.getLoc(), subtraction, subtraction);
  // - ( (x-mean)^2 / 2 * stddev^2 )
  auto fraction = rewriter.create<mlir::MulFOp>(op.getLoc(), numerator, denominatorConst);
  // -ln(stddev) - 1/2 ln(2*pi) - 1/2*(stddev^2) * (x - mean)^2
  Value gaussian = rewriter.create<mlir::AddFOp>(op->getLoc(), coefficientConst, fraction);
  if (op.supportMarginal()) {
    auto isNan = rewriter.create<mlir::CmpFOp>(op->getLoc(), CmpFPredicate::UNO, feature, feature);
    auto constOne = broadcastVectorConstant(targetVectorType, 0.0, rewriter, op.getLoc());
    gaussian = rewriter.create<mlir::SelectOp>(op.getLoc(), isNan, constOne, gaussian);
  }
  rewriter.replaceOp(op, gaussian);
  return success();
}

// Anonymous namespace holding helper functions.
namespace {

  template<typename SourceOp>
  mlir::LogicalResult replaceOpWithGatherFromGlobalMemref(SourceOp op,
                                                          mlir::ConversionPatternRewriter& rewriter,
                                                          mlir::Value indexOperand,
                                                          int lowerBound,
                                                          int upperBound,
                                                          llvm::ArrayRef<mlir::Attribute> arrayValues,
                                                          mlir::Type resultType,
                                                          const std::string& tablePrefix, bool computesLog) {
    static int tableCount = 0;
    auto inputType = indexOperand.getType();
    if (!inputType.template isa<mlir::VectorType>()) {
      // This pattern only handles vectorized implementations and fails if the input is not a vector.
      return rewriter.notifyMatchFailure(op, "Match failed because input is not a VectorType");
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
    auto symbolName = tablePrefix + std::to_string(tableCount++);
    auto visibility = rewriter.getStringAttr("private");
    auto memrefType = mlir::MemRefType::get({(long) arrayValues.size()}, resultType);
    (void) rewriter.create<mlir::memref::GlobalOp>(op.getLoc(), symbolName, visibility,
                                                   memrefType, valArrayAttr, true);
    // Restore insertion point
    rewriter.restoreInsertionPoint(restore);

    // Use GetGlobalMemref operation to access the global created above.
    auto addressOf = rewriter.template create<mlir::memref::GetGlobalOp>(op.getLoc(), memrefType, symbolName);
    auto vectorShape = inputType.template dyn_cast<mlir::VectorType>().getShape();
    // Convert the input to integer type if necessary.
    mlir::Value index = indexOperand;
    auto indexType = inputType.template dyn_cast<mlir::VectorType>().getElementType();
    if (!indexType.isIntOrIndex()) {
      if (indexType.template isa<mlir::FloatType>()) {
        indexType = rewriter.getI64Type();
        index = rewriter.template create<mlir::FPToUIOp>(op.getLoc(), index,
                                                         mlir::VectorType::get(vectorShape, indexType));
      } else {
        // The input type is neither int/index nor float, conversion unknown, fail this pattern.
        return rewriter.notifyMatchFailure(op, "Match failed because input is neither float nor integer/index");
      }
    }
    // Construct the constant pass-thru value (values used if the mask is false for an element of the vector).
    auto vectorType = mlir::VectorType::get(vectorShape, resultType);
    double defaultValue = (computesLog) ? static_cast<double>(-INFINITY) : 0.0;
    auto passThru = broadcastVectorConstant(vectorType, defaultValue,
                                            rewriter, op->getLoc());

    mlir::ConstantOp idxMinVec;
    mlir::ConstantOp idxMaxVec;
    if (indexType.isInteger(32)) {
      idxMinVec =
          broadcastVectorConstant(mlir::VectorType::get(vectorShape, indexType), lowerBound, rewriter, op->getLoc());
      idxMaxVec =
          broadcastVectorConstant(mlir::VectorType::get(vectorShape, indexType), upperBound, rewriter, op->getLoc());
    } else {
      idxMinVec = broadcastVectorConstant(mlir::VectorType::get(vectorShape, indexType),
                                          static_cast<long>(lowerBound),
                                          rewriter,
                                          op->getLoc());
      idxMaxVec = broadcastVectorConstant(mlir::VectorType::get(vectorShape, indexType),
                                          static_cast<long>(upperBound),
                                          rewriter,
                                          op->getLoc());
    }

    auto boolVecTy = mlir::VectorType::get(vectorShape, rewriter.getI1Type());
    auto inBoundsLB =
        rewriter.create<mlir::CmpIOp>(op->getLoc(), boolVecTy, mlir::CmpIPredicate::sge, index, idxMinVec);
    auto inBoundsUB =
        rewriter.create<mlir::CmpIOp>(op->getLoc(), boolVecTy, mlir::CmpIPredicate::slt, index, idxMaxVec);
    auto inBounds = rewriter.create<mlir::AndOp>(op->getLoc(), inBoundsLB, inBoundsUB);
    auto trueVec =
        broadcastVectorConstant(mlir::VectorType::get(vectorShape, rewriter.getI1Type()), true, rewriter, op->getLoc());
    auto falseVec = broadcastVectorConstant(mlir::VectorType::get(vectorShape, rewriter.getI1Type()),
                                            false,
                                            rewriter,
                                            op->getLoc());
    // Create mask, dependant on an "inBounds" select -> Mask all out-of-bounds accesses.
    auto mask = rewriter.create<mlir::SelectOp>(op.getLoc(), inBounds, trueVec, falseVec);

    // Replace the source operation with a gather load from the global memref.
    mlir::Value constIndex = rewriter.template create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
    mlir::Value leaf = rewriter.template create<mlir::vector::GatherOp>(op.getLoc(), vectorType, addressOf,
                                                                        constIndex, index, mask, passThru);
    if (op.supportMarginal()) {
      assert(indexType.template isa<mlir::FloatType>());
      auto isNan = rewriter.create<mlir::CmpFOp>(op->getLoc(), mlir::CmpFPredicate::UNO,
                                                 indexOperand, indexOperand);
      auto marginalValue = (computesLog) ? 0.0 : 1.0;
      auto constOne = broadcastVectorConstant(vectorType, marginalValue, rewriter, op.getLoc());
      leaf = rewriter.create<mlir::SelectOp>(op.getLoc(), isNan, constOne, leaf);
    }
    rewriter.replaceOp(op, leaf);
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
  Type resultType = op.getResult().getType();
  bool computesLog = false;
  if (auto logType = resultType.dyn_cast<low::LogType>()) {
    resultType = logType.getBaseType();
    computesLog = true;
  }
  SmallVector<Attribute, 5> values;
  for (auto val : op.probabilities().getValue()) {
    if (computesLog) {
      auto floatVal = val.dyn_cast<FloatAttr>();
      assert(floatVal);
      values.push_back(FloatAttr::get(resultType, log(floatVal.getValueAsDouble())));
    } else {
      values.push_back(val);
    }
  }
  int idxMax = op.probabilities().size();
  return replaceOpWithGatherFromGlobalMemref<low::SPNCategoricalLeaf>(op, rewriter, operands[0], 0, idxMax,
                                                                      values, resultType, "categorical_vec_",
                                                                      computesLog);
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
  for (auto& b: op.bucketsAttr()) {
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

  Type resultType = op.getResult().getType();
  bool computesLog = false;
  if (auto logType = resultType.dyn_cast<low::LogType>()) {
    resultType = logType.getBaseType();
    computesLog = true;
  }
  if (!resultType.isIntOrFloat()) {
    // Currently only handling Int and Float result types.
    return failure();
  }

  // Flatten the map into an array by filling up empty indices with 0 values.
  SmallVector<Attribute, 256> valArray;
  for (int i = 0; i < maxUB; ++i) {
    double indexVal = NAN;
    if (values.count(i)) {
      indexVal = (computesLog) ? log(values[i]) : values[i];
    } else {
      // Fill up with 0 if no value was defined by the histogram.
      indexVal = (computesLog) ? static_cast<double>(-INFINITY) : 0;
    }
    // Construct attribute with constant value. Need to distinguish cases here due to different builder methods.
    if (resultType.isIntOrIndex()) {
      valArray.push_back(rewriter.getIntegerAttr(resultType, (int) indexVal));
    } else {
      valArray.push_back(rewriter.getFloatAttr(resultType, indexVal));
    }

  }
  return replaceOpWithGatherFromGlobalMemref<low::SPNHistogramLeaf>(op, rewriter, operands[0], minLB, maxUB, valArray,
                                                                    resultType, "histogram_vec_",
                                                                    computesLog);
}

mlir::LogicalResult mlir::spn::ResolveVectorizedStripLog::matchAndRewrite(low::SPNStripLog op,
                                                                          ArrayRef<Value> operands,
                                                                          ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only resolves vectorized operation");
  }
  assert(operands.size() == 1);
  auto vectorType = operands[0].getType().dyn_cast<VectorType>();
  if (!vectorType) {
    return rewriter.notifyMatchFailure(op, "Expected operand to have vector type");
  }
  if (vectorType.getElementType() != op.target()) {
    return rewriter.notifyMatchFailure(op, "Could not resolve StripLog trivially");
  }
  rewriter.replaceOp(op, operands[0]);
  return success();
}

mlir::LogicalResult mlir::spn::ResolveVectorizedConvertLog::matchAndRewrite(mlir::spn::low::SPNConvertLog op,
                                                                            llvm::ArrayRef<mlir::Value> operands,
                                                                            mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only resolves vectorized operation");
  }
  assert(operands.size() == 1);
  auto vectorType = operands[0].getType().dyn_cast<VectorType>();
  if (!vectorType) {
    return rewriter.notifyMatchFailure(op, "Expected operand to have vector type");
  }
  if (vectorType.getElementType() != op.getResult().getType().cast<low::LogType>().getBaseType()) {
    return rewriter.notifyMatchFailure(op, "Could not resolve ConvertLog trivially");
  }
  rewriter.replaceOp(op, operands[0]);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeSelectLeaf::matchAndRewrite(mlir::spn::low::SPNSelectLeaf op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  if (!op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches vectorized Select");
  }

  auto inputVec = operands.front();
  auto inputVecTy = inputVec.getType();

  // Input should be a vector
  if (!inputVecTy.isa<VectorType>()) {
    return rewriter.notifyMatchFailure(op, "Vectorization pattern did not match, input was not a vector");
  }

  auto inputTy = op.input().getType();
  VectorType vectorType = inputVecTy.dyn_cast<VectorType>();

  // If the input type is not an integer, but also not a float, we cannot convert it and this pattern fails.
  mlir::Value cond;
  auto booleanVectorTy = VectorType::get(vectorType.getShape(), IntegerType::get(op.getContext(), 1));
  if (inputTy.isa<mlir::FloatType>()) {
    auto thresholdVec =
        broadcastVectorConstant(vectorType, op.input_true_thresholdAttr().getValueAsDouble(), rewriter, op.getLoc());
    cond =
        rewriter.create<mlir::CmpFOp>(op->getLoc(), booleanVectorTy, mlir::CmpFPredicate::ULT, inputVec, thresholdVec);
  } else if (inputTy.isa<mlir::IntegerType>()) {
    auto thresholdVec =
        broadcastVectorConstant(vectorType, op.input_true_thresholdAttr().getValueAsDouble(), rewriter, op.getLoc());
    cond =
        rewriter.create<mlir::CmpIOp>(op->getLoc(), booleanVectorTy, mlir::CmpIPredicate::ult, inputVec, thresholdVec);
  } else {
    return rewriter.notifyMatchFailure(op, "Expected condition-value to be either Float- or IntegerType");
  }

  Type resultType = op.getResult().getType();
  bool computesLog = false;
  if (auto logType = resultType.dyn_cast<low::LogType>()) {
    resultType = logType.getBaseType();
    computesLog = true;
  }

  ConstantOp val_true, val_false;
  if (computesLog) {
    auto logVecTy = VectorType::get(vectorType.getShape(), resultType);
    val_true = broadcastVectorConstant(logVecTy, log(op.val_trueAttr().getValueAsDouble()), rewriter, op.getLoc());
    val_false = broadcastVectorConstant(logVecTy, log(op.val_falseAttr().getValueAsDouble()), rewriter, op.getLoc());
  } else {
    val_true = broadcastVectorConstant(vectorType, op.val_trueAttr().getValueAsDouble(), rewriter, op.getLoc());
    val_false = broadcastVectorConstant(vectorType, op.val_falseAttr().getValueAsDouble(), rewriter, op.getLoc());
  }

  mlir::Value leaf = rewriter.create<SelectOp>(op.getLoc(), cond, val_true, val_false);
  if (op.supportMarginal()) {
    assert(inputTy.isa<mlir::FloatType>());
    auto isNan = rewriter.create<mlir::CmpFOp>(op->getLoc(), CmpFPredicate::UNO, inputVec, inputVec);
    auto marginalValue = (computesLog) ? 0.0 : 1.0;
    auto constOne = broadcastVectorConstant(vectorType, marginalValue, rewriter, op.getLoc());
    leaf = rewriter.create<mlir::SelectOp>(op.getLoc(), isNan, constOne, leaf);
  }
  rewriter.replaceOp(op, leaf);

  return success();
}