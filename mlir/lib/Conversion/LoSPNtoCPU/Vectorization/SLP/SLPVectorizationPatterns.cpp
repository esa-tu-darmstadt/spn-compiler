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

// === SLPPatternApplicator === //

SLPPatternApplicator::SLPPatternApplicator(SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>&& patterns)
    : patterns{std::move(patterns)} {
  // Patterns with higher benefit should always be applied first.
  llvm::sort(std::begin(this->patterns), std::end(this->patterns), [&](auto const& lhs, auto const& rhs) {
    return lhs->getBenefit() > rhs->getBenefit();
  });
}

SLPVectorizationPattern* SLPPatternApplicator::bestMatch(ValueVector* vector) {
  auto it = bestMatches.try_emplace(vector, nullptr);
  if (it.second) {
    for (auto const& pattern : patterns) {
      if (succeeded(pattern->matchVector(vector))) {
        it.first->getSecond() = pattern.get();
      }
    }
  }
  return it.first->second;
}

LogicalResult SLPPatternApplicator::matchAndRewrite(ValueVector* vector, PatternRewriter& rewriter) {
  auto* pattern = bestMatch(vector);
  if (!pattern) {
    return failure();
  }
  pattern->rewriteVector(vector, rewriter);
  bestMatches.erase(vector);
  return success();
}

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

// === SPNConstant === //

unsigned VectorizeConstant::costIfMatches(ValueVector* vector) const {
  return 0;
}

void VectorizeConstant::rewrite(ValueVector* vector, PatternRewriter& rewriter) const {
  SmallVector<Attribute, 4> constants;
  for (auto const& value : *vector) {
    constants.emplace_back(static_cast<ConstantOp>(value.getDefiningOp()).value());
  }
  auto const& elements = denseFloatingPoints(std::begin(constants), std::end(constants), vector->getVectorType());
  auto const& constVector = conversionManager.getOrCreateConstant(vector->getLoc(), elements);
  conversionManager.update(vector, constVector, ElementFlag::KeepNoneNoExtract);
}

// === SPNBatchRead === //

LogicalResult VectorizeBatchRead::matchVector(ValueVector* vector) const {
  if (!consecutiveLoads(vector->begin(), vector->end())) {
    // Pattern only applicable to consecutive loads.
    return failure();
  }
  return success();
}

void VectorizeBatchRead::rewrite(ValueVector* vector, PatternRewriter& rewriter) const {
  auto batchRead = cast<SPNBatchRead>(vector->getElement(0).getDefiningOp());
  auto sampleIndex =
      conversionManager.getOrCreateConstant(vector->getLoc(), rewriter.getIndexAttr(batchRead.sampleIndex()));
  ValueRange indices{batchRead.batchIndex(), sampleIndex};
  auto vectorLoad =
      rewriter.create<vector::LoadOp>(vector->getLoc(), vector->getVectorType(), batchRead.batchMem(), indices);
  conversionManager.update(vector, vectorLoad, ElementFlag::KeepNone);
}

// === SPNAdd === //

void VectorizeAdd::rewrite(ValueVector* vector, PatternRewriter& rewriter) const {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < vector->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(vector->getOperand(i)));
  }
  auto vectorAdd = rewriter.create<AddFOp>(vector->getLoc(), vector->getVectorType(), operands);
  conversionManager.update(vector, vectorAdd, ElementFlag::KeepNone);
}

// === SPNMul === //

void VectorizeMul::rewrite(ValueVector* vector, PatternRewriter& rewriter) const {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < vector->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(vector->getOperand(i)));
  }
  auto vectorAdd = rewriter.create<MulFOp>(vector->getLoc(), vector->getVectorType(), operands);
  conversionManager.update(vector, vectorAdd, ElementFlag::KeepNone);
}

// === SPNGaussianLeaf === //

unsigned VectorizeGaussian::costIfMatches(ValueVector* vector) const {
  return 6;
}

void VectorizeGaussian::rewrite(ValueVector* vector, PatternRewriter& rewriter) const {

  auto vectorType = vector->getVectorType();

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
  // (x - mean)
  auto meanVector = conversionManager.getOrCreateConstant(vector->getLoc(), means);
  Value gaussianVector = rewriter.create<SubFOp>(vector->getLoc(), vectorType, inputVector, meanVector);

  // ((x - mean) / stddev)^2
  auto stddevVector = conversionManager.getOrCreateConstant(vector->getLoc(), stddevs);
  gaussianVector = rewriter.create<DivFOp>(vector->getLoc(), vectorType, gaussianVector, stddevVector);
  gaussianVector = rewriter.create<MulFOp>(vector->getLoc(), vectorType, gaussianVector, gaussianVector);

  // e^(-0.5 * ((x - mean) / stddev)^2))
  auto halfVector = conversionManager.getOrCreateConstant(vector->getLoc(), denseFloatingPoints(-0.5, vectorType));
  gaussianVector = rewriter.create<MulFOp>(vector->getLoc(), vectorType, halfVector, gaussianVector);
  gaussianVector = rewriter.create<math::ExpOp>(vector->getLoc(), vectorType, gaussianVector);

  // e^(-0.5 * ((x - mean) / stddev)^2)) / (stddev * sqrt(2 * PI))
  auto coefficientVector = conversionManager.getOrCreateConstant(vector->getLoc(), coefficients);
  gaussianVector = rewriter.create<MulFOp>(vector->getLoc(), coefficientVector, gaussianVector);

  conversionManager.update(vector, gaussianVector, ElementFlag::KeepNone);
}
