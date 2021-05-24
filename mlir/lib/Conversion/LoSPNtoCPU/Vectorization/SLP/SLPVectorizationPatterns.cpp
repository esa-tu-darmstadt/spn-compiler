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
    return lhs->getCost() < rhs->getCost();
  });
}

SLPVectorizationPattern* SLPPatternApplicator::bestMatch(Superword* vector) {
  auto it = bestMatches.try_emplace(vector, nullptr);
  if (it.second) {
    for (auto const& pattern : patterns) {
      if (succeeded(pattern->matchSuperword(vector))) {
        it.first->getSecond() = pattern.get();
      }
    }
  }
  return it.first->second;
}

LogicalResult SLPPatternApplicator::matchAndRewrite(Superword* vector, PatternRewriter& rewriter) {
  auto* pattern = bestMatch(vector);
  if (!pattern) {
    return failure();
  }
  pattern->rewriteSuperword(vector, rewriter);
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

unsigned VectorizeConstant::getCost() const {
  return 0;
}

void VectorizeConstant::rewrite(Superword* superword, PatternRewriter& rewriter) const {
  SmallVector<Attribute, 4> constants;
  for (auto const& value : *superword) {
    constants.emplace_back(static_cast<ConstantOp>(value.getDefiningOp()).value());
  }
  auto const& elements = denseFloatingPoints(std::begin(constants), std::end(constants), superword->getVectorType());
  auto const& constVector = conversionManager.getOrCreateConstant(superword->getLoc(), elements);
  conversionManager.update(superword, constVector, ElementFlag::KeepNoneNoExtract);
}

// === SPNBatchRead === //

LogicalResult VectorizeBatchRead::matchSuperword(Superword* superword) const {
  if (!consecutiveLoads(superword->begin(), superword->end())) {
    // Pattern only applicable to consecutive loads.
    return failure();
  }
  return success();
}

void VectorizeBatchRead::rewrite(Superword* superword, PatternRewriter& rewriter) const {
  auto batchRead = cast<SPNBatchRead>(superword->getElement(0).getDefiningOp());
  auto sampleIndex =
      conversionManager.getOrCreateConstant(superword->getLoc(), rewriter.getIndexAttr(batchRead.sampleIndex()));
  ValueRange indices{batchRead.batchIndex(), sampleIndex};
  auto vectorLoad =
      rewriter.create<vector::LoadOp>(superword->getLoc(), superword->getVectorType(), batchRead.batchMem(), indices);
  conversionManager.update(superword, vectorLoad, ElementFlag::KeepNone);
}

// === SPNAdd === //

void VectorizeAdd::rewrite(Superword* superword, PatternRewriter& rewriter) const {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  auto vectorAdd = rewriter.create<AddFOp>(superword->getLoc(), superword->getVectorType(), operands);
  conversionManager.update(superword, vectorAdd, ElementFlag::KeepNone);
}

// === SPNMul === //

void VectorizeMul::rewrite(Superword* superword, PatternRewriter& rewriter) const {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  auto vectorAdd = rewriter.create<MulFOp>(superword->getLoc(), superword->getVectorType(), operands);
  conversionManager.update(superword, vectorAdd, ElementFlag::KeepNone);
}

// === SPNGaussianLeaf === //

unsigned VectorizeGaussian::getCost() const {
  return 6;
}

void VectorizeGaussian::rewrite(Superword* superword, PatternRewriter& rewriter) const {

  auto vectorType = superword->getVectorType();

  DenseElementsAttr coefficients;
  if (vectorType.getElementType().cast<FloatType>().getWidth() == 32) {
    SmallVector<float, 4> array;
    for (auto const& value : *superword) {
      float stddev = static_cast<SPNGaussianLeaf>(value.getDefiningOp()).stddev().convertToFloat();
      array.emplace_back(1.0f / (stddev * std::sqrt(2.0f * M_PIf32)));
    }
    coefficients = DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<float>>(array));
  } else {
    SmallVector<double, 4> array;
    for (auto const& value : *superword) {
      double stddev = static_cast<SPNGaussianLeaf>(value.getDefiningOp()).stddev().convertToDouble();
      array.emplace_back(1.0 / (stddev * std::sqrt(2.0 * M_PI)));
    }
    coefficients = DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<double>>(array));
  }

  // Gather means in a dense floating point attribute vector.
  SmallVector<Attribute, 4> meanAttributes;
  for (auto const& value : *superword) {
    meanAttributes.emplace_back(static_cast<SPNGaussianLeaf>(value.getDefiningOp()).meanAttr());
  }
  auto const& means = denseFloatingPoints(std::begin(meanAttributes), std::end(meanAttributes), vectorType);

  // Gather standard deviations in a dense floating point attribute vector.
  SmallVector<Attribute, 4> stddevAttributes;
  for (auto const& value : *superword) {
    stddevAttributes.emplace_back(static_cast<SPNGaussianLeaf>(value.getDefiningOp()).stddevAttr());
  }
  auto const& stddevs = denseFloatingPoints(std::begin(stddevAttributes), std::end(stddevAttributes), vectorType);

  // Grab the input vector.
  Value const& inputVector = conversionManager.getValue(superword->getOperand(0));

  // Calculate Gaussian distribution using e^(-0.5 * ((x - mean) / stddev)^2)) / (stddev * sqrt(2 * PI))
  // (x - mean)
  auto meanVector = conversionManager.getOrCreateConstant(superword->getLoc(), means);
  Value gaussianVector = rewriter.create<SubFOp>(superword->getLoc(), vectorType, inputVector, meanVector);

  // ((x - mean) / stddev)^2
  auto stddevVector = conversionManager.getOrCreateConstant(superword->getLoc(), stddevs);
  gaussianVector = rewriter.create<DivFOp>(superword->getLoc(), vectorType, gaussianVector, stddevVector);
  gaussianVector = rewriter.create<MulFOp>(superword->getLoc(), vectorType, gaussianVector, gaussianVector);

  // e^(-0.5 * ((x - mean) / stddev)^2))
  auto halfVector = conversionManager.getOrCreateConstant(superword->getLoc(), denseFloatingPoints(-0.5, vectorType));
  gaussianVector = rewriter.create<MulFOp>(superword->getLoc(), vectorType, halfVector, gaussianVector);
  gaussianVector = rewriter.create<math::ExpOp>(superword->getLoc(), vectorType, gaussianVector);

  // e^(-0.5 * ((x - mean) / stddev)^2)) / (stddev * sqrt(2 * PI))
  auto coefficientVector = conversionManager.getOrCreateConstant(superword->getLoc(), coefficients);
  gaussianVector = rewriter.create<MulFOp>(superword->getLoc(), coefficientVector, gaussianVector);

  conversionManager.update(superword, gaussianVector, ElementFlag::KeepNone);
}
