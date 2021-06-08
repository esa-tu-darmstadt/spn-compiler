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
  DenseMap<Value, unsigned> elementCounts;
  Value broadcastValue;
  unsigned maxCount = 0;
  for (auto const& element : *superword) {
    if (++elementCounts[element] > maxCount) {
      broadcastValue = element;
      maxCount = elementCounts[element];
    }
  }
  auto loc = superword->getLoc();
  Value vectorizedOp = rewriter.create<vector::BroadcastOp>(loc, superword->getVectorType(), broadcastValue);
  for (size_t i = 0; i < superword->numLanes(); ++i) {
    auto const& element = superword->getElement(i);
    if (element == broadcastValue) {
      continue;
    }
    auto index = conversionManager.getOrCreateConstant(loc, rewriter.getI32IntegerAttr((int) i));
    vectorizedOp = rewriter.create<vector::InsertElementOp>(loc, element, vectorizedOp, index);
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

}

// === VectorizeConstant === //

void VectorizeConstant::rewrite(Superword* superword, PatternRewriter& rewriter) const {
  SmallVector<Attribute, 4> constants;
  for (auto const& value : *superword) {
    constants.emplace_back(static_cast<ConstantOp>(value.getDefiningOp()).value());
  }
  auto const& elements = denseFloatingPoints(std::begin(constants), std::end(constants), superword->getVectorType());
  auto const& constVector = conversionManager.getOrCreateConstant(superword->getLoc(), elements);
  conversionManager.update(superword, constVector, ElementFlag::KeepAll);
}

void VectorizeConstant::accept(PatternVisitor& visitor, Superword* superword) {
  visitor.visit(this, superword);
}

// === VectorizeBatchRead === //

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

// === VectorizeAdd === //

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

// === VectorizeMul === //

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

// === VectorizeGaussian === //

void VectorizeGaussian::rewrite(Superword* superword, PatternRewriter& rewriter) const {

  auto vectorType = superword->getVectorType();

  // Calculate Gaussian distribution using e^((x - mean)^2 / (-2 * variance)) / sqrt(2 * PI * variance)
  DenseElementsAttr means;
  DenseElementsAttr variances;
  DenseElementsAttr roots;
  if (vectorType.getElementType().cast<FloatType>().getWidth() == 32) {
    SmallVector<float, 4> meansArray;
    SmallVector<float, 4> variancesArray;
    SmallVector<float, 4> rootsArray;
    for (auto const& value : *superword) {
      meansArray.emplace_back(static_cast<SPNGaussianLeaf>(value.getDefiningOp()).mean().convertToFloat());
      float stddev = static_cast<SPNGaussianLeaf>(value.getDefiningOp()).stddev().convertToFloat();
      variancesArray.emplace_back(-2 * stddev * stddev);
      rootsArray.emplace_back(std::sqrt(2.0f * M_PIf32 * stddev * stddev));
    }
    means = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<float>>(meansArray));
    variances = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<float>>(variancesArray));
    roots = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<float>>(rootsArray));
  } else {
    SmallVector<double, 4> meansArray;
    SmallVector<double, 4> variancesArray;
    SmallVector<double, 4> rootsArray;
    for (auto const& value : *superword) {
      meansArray.emplace_back(static_cast<SPNGaussianLeaf>(value.getDefiningOp()).mean().convertToDouble());
      double stddev = static_cast<SPNGaussianLeaf>(value.getDefiningOp()).stddev().convertToDouble();
      variancesArray.emplace_back(-2 * stddev * stddev);
      rootsArray.emplace_back(std::sqrt(2.0 * M_PI * stddev * stddev));
    }
    means = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(meansArray));
    variances = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(variancesArray));
    roots = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(rootsArray));
  }

  // (x - mean)
  auto inputVector = conversionManager.getValue(superword->getOperand(0));
  auto meanVector = conversionManager.getOrCreateConstant(superword->getLoc(), means);
  Value gaussianVector = rewriter.create<SubFOp>(superword->getLoc(), vectorType, inputVector, meanVector);

  // ((x - mean)^2
  gaussianVector = rewriter.create<MulFOp>(superword->getLoc(), vectorType, gaussianVector, gaussianVector);

  // (x - mean)^2 / (-2 * variance)
  auto varianceVector = conversionManager.getOrCreateConstant(superword->getLoc(), variances);
  gaussianVector = rewriter.create<DivFOp>(superword->getLoc(), vectorType, gaussianVector, varianceVector);

  // e^((x - mean)^2 / (-2 * variance))
  gaussianVector = rewriter.create<math::ExpOp>(superword->getLoc(), vectorType, gaussianVector);

  // e^((x - mean)^2 / (-2 * variance)) / sqrt(2 * PI * variance)
  auto rootsVector = conversionManager.getOrCreateConstant(superword->getLoc(), roots);
  gaussianVector = rewriter.create<DivFOp>(superword->getLoc(), gaussianVector, rootsVector);

  conversionManager.update(superword, gaussianVector, ElementFlag::KeepNone);
}

void VectorizeGaussian::accept(PatternVisitor& visitor, Superword* superword) {
  visitor.visit(this, superword);
}
