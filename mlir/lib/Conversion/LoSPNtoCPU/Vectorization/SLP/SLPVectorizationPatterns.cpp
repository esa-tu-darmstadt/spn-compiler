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

// === SLPVectorizationPattern === //

SLPVectorizationPattern::SLPVectorizationPattern(ConversionManager& conversionManager) : conversionManager{
    conversionManager} {}

void SLPVectorizationPattern::rewriteSuperword(Superword* superword, PatternRewriter& rewriter) {
  conversionManager.setInsertionPointFor(superword);
  rewrite(superword, rewriter);
}

// Helper functions in anonymous namespace.
namespace {
  void createNecessaryExtractionsFor(Superword* superword, ConversionManager& conversionManager) {
    // Create extractions from vectorized operands if present.
    for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
      auto const& element = superword->getElement(lane);
      if (auto* elementOp = element.getDefiningOp()) {
        if (superword->isLeaf()) {
          superword->setElement(lane, conversionManager.getOrExtractValue(element));
        } else {
          for (size_t i = 0; i < elementOp->getNumOperands(); ++i) {
            elementOp->setOperand(i, conversionManager.getOrExtractValue(elementOp->getOperand(i)));
          }
        }
      }
      if (lane == 0 && superword->splattable()) {
        break;
      }
    }
  }
}

// === Broadcast === //

LogicalResult BroadcastSuperword::match(Superword* superword) const {
  return success(superword->splattable());
}

void BroadcastSuperword::rewrite(Superword* superword, PatternRewriter& rewriter) const {
  createNecessaryExtractionsFor(superword, conversionManager);
  auto const& element = superword->getElement(0);
  auto vectorizedOp = rewriter.create<vector::BroadcastOp>(element.getLoc(), superword->getVectorType(), element);
  conversionManager.update(superword, vectorizedOp, ElementFlag::KeepFirst);
}

void BroadcastSuperword::accept(PatternVisitor& visitor, Superword* superword) {
  visitor.visit(this, superword);
}

// === BroadcastInsert === //

LogicalResult BroadcastInsertSuperword::match(Superword* superword) const {
  return success();
}

void BroadcastInsertSuperword::rewrite(Superword* superword, PatternRewriter& rewriter) const {
  createNecessaryExtractionsFor(superword, conversionManager);
  Value vectorizedOp;
  for (size_t i = 0; i < superword->numLanes(); ++i) {
    auto const& element = superword->getElement(i);
    if (i == 0) {
      vectorizedOp = rewriter.create<vector::BroadcastOp>(element.getLoc(), superword->getVectorType(), element);
    } else {
      auto index = conversionManager.getOrCreateConstant(element.getLoc(), rewriter.getI32IntegerAttr((int) i));
      vectorizedOp = rewriter.create<vector::InsertElementOp>(element.getLoc(), element, vectorizedOp, index);
    }
  }
  conversionManager.update(superword, vectorizedOp, ElementFlag::KeepAll);
}

void BroadcastInsertSuperword::accept(PatternVisitor& visitor, Superword* superword) {
  visitor.visit(this, superword);
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

void VectorizeConstant::rewrite(Superword* superword, PatternRewriter& rewriter) const {
  SmallVector<Attribute, 4> constants;
  for (auto const& value : *superword) {
    constants.emplace_back(static_cast<ConstantOp>(value.getDefiningOp()).value());
  }
  auto const& elements = denseFloatingPoints(std::begin(constants), std::end(constants), superword->getVectorType());
  auto const& constVector = conversionManager.getOrCreateConstant(superword->getLoc(), elements);
  conversionManager.update(superword, constVector, ElementFlag::KeepNoneNoExtract);
}

void VectorizeConstant::accept(PatternVisitor& visitor, Superword* superword) {
  visitor.visit(this, superword);
}

// === SPNBatchRead === //

LogicalResult VectorizeBatchRead::match(Superword* superword) const {
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

void VectorizeBatchRead::accept(PatternVisitor& visitor, Superword* superword) {
  visitor.visit(this, superword);
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

void VectorizeAdd::accept(PatternVisitor& visitor, Superword* superword) {
  visitor.visit(this, superword);
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

void VectorizeMul::accept(PatternVisitor& visitor, Superword* superword) {
  visitor.visit(this, superword);
}

// === SPNGaussianLeaf === //

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

void VectorizeGaussian::accept(PatternVisitor& visitor, Superword* superword) {
  visitor.visit(this, superword);
}
