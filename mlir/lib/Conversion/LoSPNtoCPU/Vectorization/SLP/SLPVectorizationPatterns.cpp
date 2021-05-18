//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

// Helper functions in anonymous namespace.
namespace {

  template<typename AttributeIterator>
  DenseElementsAttr denseFloatingPoints(AttributeIterator begin, AttributeIterator end, VectorType const& vectorType) {
    if (vectorType.getElementType().cast<FloatType>().getWidth() == 32) {
      SmallVector<float, 4> array;
      while (begin != end) {
        array.push_back(begin->template cast<FloatAttr>().getValue().convertToFloat());
        ++begin;
      }
      return DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<float>>(array));
    }
    SmallVector<double, 4> array;
    while (begin != end) {
      array.push_back(begin->template cast<FloatAttr>().getValue().convertToDouble());
      ++begin;
    }
    return DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<double>>(array));
  }

  template<typename T>
  DenseElementsAttr denseFloatingPoints(T value, VectorType const& vectorType) {
    if (vectorType.getElementType().cast<FloatType>().getWidth() == 32) {
      SmallVector<float, 4> array;
      for (unsigned i = 0; i < vectorType.getNumElements(); ++i) {
        array.push_back(static_cast<float>(value));
      }
      return DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<float>>(array));
    }
    SmallVector<double, 4> array;
    for (unsigned i = 0; i < vectorType.getNumElements(); ++i) {
      array.push_back(value);
    }
    return DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<double>>(array));
  }

}

LogicalResult VectorizeConstant::matchAndRewrite(ConstantOp constantOp, PatternRewriter& rewriter) const {

  SmallVector<Attribute, 4> constants;
  for (auto const& value : *vector) {
    if (auto definingOp = value.getDefiningOp<ConstantOp>()) {
      constants.emplace_back(definingOp.value());
    } else {
      return rewriter.notifyMatchFailure(constantOp, "Pattern only applicable to uniform constant vectors.");
    }
  }

  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), constantOp.getType());
  conversionManager.setInsertionPointFor(vector);

  auto const& elements = denseFloatingPoints(std::begin(constants), std::end(constants), vectorType);
  auto const& constVector = conversionManager.getOrCreateConstant(constantOp.getLoc(), elements);

  conversionManager.update(vector, constVector, ElementFlag::KeepNoneNoExtract);

  return success();
}

LogicalResult VectorizeBatchRead::matchAndRewrite(SPNBatchRead batchReadOp, PatternRewriter& rewriter) const {

  if (!consecutiveLoads(vector->begin(), vector->end())) {
    return rewriter.notifyMatchFailure(batchReadOp, "Pattern only applicable to consecutive loads.");
  }

  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), batchReadOp.getType());
  conversionManager.setInsertionPointFor(vector);

  auto loc = batchReadOp.getLoc();
  auto sampleIndex = conversionManager.getOrCreateConstant(loc, rewriter.getIndexAttr(batchReadOp.sampleIndex()));
  ValueRange indices{batchReadOp.batchIndex(), sampleIndex};
  auto vectorLoad = rewriter.create<vector::LoadOp>(loc, vectorType, batchReadOp.batchMem(), indices);
  conversionManager.update(vector, vectorLoad, ElementFlag::KeepNone);

  return success();
}

LogicalResult VectorizeAdd::matchAndRewrite(SPNAdd addOp, PatternRewriter& rewriter) const {

  if (!vector->uniform()) {
    return rewriter.notifyMatchFailure(addOp, "Pattern not applicable to non-uniform vectors.");
  }

  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), addOp.getType());
  conversionManager.setInsertionPointFor(vector);

  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < addOp.getNumOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(vector->getOperand(i)));
  }

  auto vectorAdd = rewriter.create<AddFOp>(addOp.getLoc(), vectorType, operands);
  conversionManager.update(vector, vectorAdd, ElementFlag::KeepNone);

  return success();
}

LogicalResult VectorizeMul::matchAndRewrite(SPNMul mulOp, PatternRewriter& rewriter) const {

  if (!vector->uniform()) {
    return rewriter.notifyMatchFailure(mulOp, "Pattern not applicable to non-uniform vectors.");
  }

  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), mulOp.getType());
  conversionManager.setInsertionPointFor(vector);

  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < mulOp.getNumOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(vector->getOperand(i)));
  }

  auto vectorAdd = rewriter.create<MulFOp>(mulOp.getLoc(), vectorType, operands);
  conversionManager.update(vector, vectorAdd, ElementFlag::KeepNone);

  return success();
}

LogicalResult VectorizeGaussian::matchAndRewrite(SPNGaussianLeaf gaussianOp, PatternRewriter& rewriter) const {

  if (!vector->uniform()) {
    return rewriter.notifyMatchFailure(gaussianOp, "Pattern not applicable to non-uniform vectors.");
  }

  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), gaussianOp.getType());
  conversionManager.setInsertionPointFor(vector);

  DenseElementsAttr coefficients;
  if (vectorType.getElementType().cast<FloatType>().getWidth() == 32) {
    SmallVector<float, 4> array;
    for (auto const& value : *vector) {
      float stddev = static_cast<SPNGaussianLeaf>(value.getDefiningOp()).stddev().convertToFloat();
      array.emplace_back(1.0f / (stddev * std::sqrt(2.0f * M_PIf32)));
    }
    coefficients = DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<float>>(array));
  } else {
    SmallVector<double, 4> array;
    for (auto const& value : *vector) {
      double stddev = static_cast<SPNGaussianLeaf>(value.getDefiningOp()).stddev().convertToDouble();
      array.emplace_back(1.0 / (stddev * std::sqrt(2.0 * M_PI)));
    }
    coefficients = DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<double>>(array));
  }

  // Gather means in a dense floating point attribute vector.
  SmallVector<Attribute, 4> meanAttributes;
  for (auto const& value : *vector) {
    meanAttributes.emplace_back(static_cast<SPNGaussianLeaf>(value.getDefiningOp()).meanAttr());
  }
  auto const& means = denseFloatingPoints(std::begin(meanAttributes), std::end(meanAttributes), vectorType);

  // Gather standard deviations in a dense floating point attribute vector.
  SmallVector<Attribute, 4> stddevAttributes;
  for (auto const& value : *vector) {
    stddevAttributes.emplace_back(static_cast<SPNGaussianLeaf>(value.getDefiningOp()).stddevAttr());
  }
  auto const& stddevs = denseFloatingPoints(std::begin(stddevAttributes), std::end(stddevAttributes), vectorType);

  // Grab the input vector.
  Value const& inputVector = conversionManager.getValue(vector->getOperand(0));

  // Calculate Gaussian distribution using e^(-0.5 * ((x - mean) / stddev)^2)) / (stddev * sqrt(2 * PI))
  auto const& gaussianLoc = gaussianOp.getLoc();

  // (x - mean)
  auto meanVector = conversionManager.getOrCreateConstant(gaussianLoc, means);
  Value gaussianVector = rewriter.create<SubFOp>(gaussianLoc, vectorType, inputVector, meanVector);

  // ((x - mean) / stddev)^2
  auto stddevVector = conversionManager.getOrCreateConstant(gaussianLoc, stddevs);
  gaussianVector = rewriter.create<DivFOp>(gaussianLoc, vectorType, gaussianVector, stddevVector);
  gaussianVector = rewriter.create<MulFOp>(gaussianLoc, vectorType, gaussianVector, gaussianVector);

  // e^(-0.5 * ((x - mean) / stddev)^2))
  auto halfVector = conversionManager.getOrCreateConstant(gaussianLoc, denseFloatingPoints(-0.5, vectorType));
  gaussianVector = rewriter.create<MulFOp>(gaussianLoc, vectorType, halfVector, gaussianVector);
  gaussianVector = rewriter.create<math::ExpOp>(gaussianLoc, vectorType, gaussianVector);

  // e^(-0.5 * ((x - mean) / stddev)^2)) / (stddev * sqrt(2 * PI))
  auto coefficientVector = conversionManager.getOrCreateConstant(gaussianLoc, coefficients);
  gaussianVector = rewriter.create<MulFOp>(gaussianLoc, coefficientVector, gaussianVector);

  conversionManager.update(vector, gaussianVector, ElementFlag::KeepNone);

  return success();
}
