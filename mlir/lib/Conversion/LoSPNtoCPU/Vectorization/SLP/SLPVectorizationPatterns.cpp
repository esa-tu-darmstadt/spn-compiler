//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/Support/FormatVariadic.h"

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

  assert(!conversionManager.wasConverted(vector) && "vector has already been created");
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), constantOp.getType());
  rewriter.setInsertionPointAfterValue(conversionManager.getInsertionPoint(vector));

  SmallVector<Attribute, 4> constants;
  for (auto const& value : *vector) {
    constants.emplace_back(value.getDefiningOp<ConstantOp>().value());
  }

  auto const& elements = denseFloatingPoints(std::begin(constants), std::end(constants), vectorType);
  auto constVector = rewriter.create<mlir::ConstantOp>(constantOp.getLoc(), elements);

  conversionManager.update(vector, constVector, ElementFlag::NoExtract);

  return success();
}

LogicalResult VectorizeBatchRead::matchAndRewrite(SPNBatchRead batchReadOp, PatternRewriter& rewriter) const {

  assert(!conversionManager.wasConverted(vector) && "vector has already been created");
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), batchReadOp.getType());
  rewriter.setInsertionPointAfterValue(conversionManager.getInsertionPoint(vector));

  if (!consecutiveLoads(vector->begin(), vector->end())) {
    if (vector->splattable()) {
      auto const& element = vector->getElement(0);
      auto vectorOperation = rewriter.create<vector::BroadcastOp>(batchReadOp.getLoc(), vectorType, element);
      conversionManager.update(vector, vectorOperation, ElementFlag::KeepFirst);
    } else {
      auto vectorOperation = broadcastFirstInsertRest(vector->begin(), vector->end(), vectorType, rewriter);
      conversionManager.update(vector, vectorOperation, ElementFlag::KeepAll);
    }
  } else {
    auto batchReadLoc = batchReadOp.getLoc();
    auto memIndex = rewriter.create<ConstantOp>(batchReadLoc, rewriter.getIndexAttr(batchReadOp.sampleIndex()));
    ValueRange indices{batchReadOp.batchIndex(), memIndex};
    auto vectorLoad = rewriter.create<vector::LoadOp>(batchReadLoc, vectorType, batchReadOp.batchMem(), indices);
    conversionManager.update(vector, vectorLoad, ElementFlag::KeepNone);
  }

  assert(conversionManager.wasConverted(vector));

  return success();
}

LogicalResult VectorizeAdd::matchAndRewrite(SPNAdd addOp, PatternRewriter& rewriter) const {

  assert(!conversionManager.wasConverted(vector) && "vector has already been created");
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), addOp.getType());
  rewriter.setInsertionPointAfterValue(conversionManager.getInsertionPoint(vector));

  if (vector->isLeaf()) {
    assert(!vector->splattable() && "addition vector should not be a leaf vector if it is splattable");
    auto vectorOperation = broadcastFirstInsertRest(vector->begin(), vector->end(), vectorType, rewriter);
    conversionManager.update(vector, vectorOperation, ElementFlag::KeepAll);
    return success();
  }

  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < addOp.getNumOperands(); ++i) {
    auto* operand = vector->getOperand(i);
    assert(conversionManager.wasConverted(operand) && "operand has not yet been vectorized");
    operands.emplace_back(conversionManager.getValue(operand));
  }

  auto vectorAdd = rewriter.create<AddFOp>(addOp.getLoc(), vectorType, operands);
  conversionManager.update(vector, vectorAdd, ElementFlag::KeepNone);

  return success();
}

LogicalResult VectorizeMul::matchAndRewrite(SPNMul mulOp, PatternRewriter& rewriter) const {

  assert(!conversionManager.wasConverted(vector) && "vector has already been created");
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), mulOp.getType());
  rewriter.setInsertionPointAfterValue(conversionManager.getInsertionPoint(vector));

  if (vector->isLeaf()) {
    assert(!vector->splattable() && "multiplication vector should not be a leaf vector if it is splattable");
    auto vectorOperation = broadcastFirstInsertRest(vector->begin(), vector->end(), vectorType, rewriter);
    conversionManager.update(vector, vectorOperation, ElementFlag::KeepAll);
    return success();
  }

  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < mulOp.getNumOperands(); ++i) {
    auto* operand = vector->getOperand(i);
    assert(conversionManager.wasConverted(operand) && "operand has not yet been vectorized");
    operands.emplace_back(conversionManager.getValue(operand));
  }

  auto vectorAdd = rewriter.create<MulFOp>(mulOp.getLoc(), vectorType, operands);
  conversionManager.update(vector, vectorAdd, ElementFlag::KeepNone);

  return success();
}

LogicalResult VectorizeGaussian::matchAndRewrite(SPNGaussianLeaf gaussianOp, PatternRewriter& rewriter) const {

  assert(!conversionManager.wasConverted(vector) && "vector has already been created");
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), gaussianOp.getType());
  rewriter.setInsertionPointAfterValue(conversionManager.getInsertionPoint(vector));

  if (vector->isLeaf()) {
    assert(!vector->splattable() && "gaussian vector should not be a leaf vector if it is splattable");
    auto vectorOperation = broadcastFirstInsertRest(vector->begin(), vector->end(), vectorType, rewriter);
    conversionManager.update(vector, vectorOperation, ElementFlag::KeepAll);
    return success();
  }

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
  assert(conversionManager.wasConverted(vector->getOperand(0)) && "input vector has not yet been vectorized");
  Value const& inputVector = conversionManager.getValue(vector->getOperand(0));

  // Calculate Gaussian distribution using e^(-0.5 * ((x - mean) / stddev)^2)) / (stddev * sqrt(2 * PI))
  auto const& gaussianLoc = gaussianOp.getLoc();

  // (x - mean)
  auto meanVector = rewriter.create<ConstantOp>(gaussianLoc, means);
  Value gaussianVector = rewriter.create<SubFOp>(gaussianLoc, vectorType, inputVector, meanVector);

  // ((x - mean) / stddev)^2
  auto stddevVector = rewriter.create<ConstantOp>(gaussianLoc, stddevs);
  gaussianVector = rewriter.create<DivFOp>(gaussianLoc, vectorType, gaussianVector, stddevVector);
  gaussianVector = rewriter.create<MulFOp>(gaussianLoc, vectorType, gaussianVector, gaussianVector);

  // e^(-0.5 * ((x - mean) / stddev)^2))
  auto halfVector = rewriter.create<ConstantOp>(gaussianLoc, denseFloatingPoints(-0.5, vectorType));
  gaussianVector = rewriter.create<MulFOp>(gaussianLoc, vectorType, halfVector, gaussianVector);
  gaussianVector = rewriter.create<math::ExpOp>(gaussianLoc, vectorType, gaussianVector);

  // e^(-0.5 * ((x - mean) / stddev)^2)) / (stddev * sqrt(2 * PI))
  auto coefficientVector = rewriter.create<ConstantOp>(gaussianLoc, coefficients);
  gaussianVector = rewriter.create<MulFOp>(gaussianLoc, coefficientVector, gaussianVector);

  conversionManager.update(vector, gaussianVector, ElementFlag::KeepNone);

  return success();
}
