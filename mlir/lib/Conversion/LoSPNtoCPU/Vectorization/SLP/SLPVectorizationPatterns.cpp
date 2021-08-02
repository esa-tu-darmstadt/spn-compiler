//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/SLP/GraphConversion.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include "LoSPNtoCPU/Vectorization/Util.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/VectorOps.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

// === SLPVectorizationPattern === //

SLPVectorizationPattern::SLPVectorizationPattern(ConversionManager& conversionManager) : conversionManager{
    conversionManager} {}

void SLPVectorizationPattern::rewriteSuperword(Superword* superword, RewriterBase& rewriter) {
  conversionManager.setupConversionFor(superword, this);
  auto vectorOp = rewrite(superword, rewriter);
  conversionManager.update(superword, vectorOp, this);
}

// Helper functions in anonymous namespace.
namespace {

  Value stripLogOrValue(Value const& value, RewriterBase& rewriter) {
    if (auto logType = value.getType().dyn_cast<LogType>()) {
      return rewriter.create<SPNStripLog>(value.getLoc(), value, logType.getBaseType());
    }
    return value;
  }

  /// Might be useful in the future.
  Value castToFloatOrValue(Value const& value, FloatType targetType, RewriterBase& rewriter) {
    if (auto floatType = value.getType().dyn_cast<FloatType>()) {
      if (floatType.getWidth() < targetType.getWidth()) {
        return rewriter.create<FPExtOp>(value.getLoc(), value, targetType);
      } else if (floatType.getWidth() > targetType.getWidth()) {
        return rewriter.create<FPTruncOp>(value.getLoc(), value, targetType);
      } else {
        return value;
      }
    } else if (auto intType = value.getType().dyn_cast<IntegerType>()) {
      if (intType.isSigned()) {
        return rewriter.create<SIToFPOp>(value.getLoc(), value, targetType);
      }
      return rewriter.create<UIToFPOp>(value.getLoc(), value, targetType);
    } else if (value.getType().isa<IndexType>()) {
      auto valueAsInt = rewriter.create<IndexCastOp>(value.getLoc(), rewriter.getI64Type(), value);
      return rewriter.create<UIToFPOp>(value.getLoc(), valueAsInt, targetType);
    }
    llvm_unreachable("value cannot be cast to float");
  }
}

// === Broadcast === //

LogicalResult BroadcastSuperword::match(Superword* superword) {
  Operation* firstOp = nullptr;
  for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
    if (superword->hasAlteredSemanticsInLane(lane)) {
      return failure();
    }
    if (auto* definingOp = superword->getElement(lane).getDefiningOp()) {
      if (lane == 0) {
        firstOp = definingOp;
        continue;
      }
      if (!OperationEquivalence::isEquivalentTo(definingOp, firstOp)) {
        return failure();
      }
    } else if (firstOp || superword->getElement(lane) != superword->getElement(0)) {
      return failure();
    }
  }
  return success();
}

Value BroadcastSuperword::rewrite(Superword* superword, RewriterBase& rewriter) {
  auto const& element = stripLogOrValue(superword->getElement(0), rewriter);
  return rewriter.create<vector::BroadcastOp>(element.getLoc(), superword->getVectorType(), element);
}

void BroadcastSuperword::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}

// === BroadcastInsert === //

LogicalResult BroadcastInsertSuperword::match(Superword* superword) {
  for (unsigned lane = 0; lane < superword->numLanes(); ++lane) {
    if (superword->hasAlteredSemanticsInLane(lane)) {
      return failure();
    }
  }
  return success();
}

Value BroadcastInsertSuperword::rewrite(Superword* superword, RewriterBase& rewriter) {
  DenseMap<Value, unsigned> elementCounts;
  Value broadcastVal;
  unsigned maxCount = 0;
  for (auto const& element : *superword) {
    if (++elementCounts[element] > maxCount) {
      broadcastVal = element;
      maxCount = elementCounts[element];
    }
  }
  broadcastVal = stripLogOrValue(broadcastVal, rewriter);

  Value vectorOp = rewriter.create<vector::BroadcastOp>(superword->getLoc(), superword->getVectorType(), broadcastVal);
  for (size_t i = 0; i < superword->numLanes(); ++i) {
    auto element = superword->getElement(i);
    if (element == broadcastVal) {
      continue;
    }
    auto index = conversionManager.getOrCreateConstant(superword->getLoc(), rewriter.getI32IntegerAttr((int) i));
    element = stripLogOrValue(element, rewriter);
    vectorOp = rewriter.create<vector::InsertElementOp>(superword->getLoc(), element, vectorOp, index);
  }
  return vectorOp;
}

void BroadcastInsertSuperword::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}

// === ShuffleTwoSuperwords === //

ShuffleTwoSuperwords::ShuffleTwoSuperwords(ConversionManager& conversionManager) : SLPVectorizationPattern(
    conversionManager) {
  conversionManager.getConversionState().addVectorCallbacks(
      [&](Superword* superword) {
        if (superword->constant()) {
          return;
        }
        for (unsigned lane = 0; lane < superword->numLanes(); ++lane) {
          if (!superword->hasAlteredSemanticsInLane(lane)) {
            superwordsByValue[superword->getElement(lane)].insert(superword);
          }
        }
      }, [&](Superword* superword) {
        for (auto element : *superword) {
          superwordsByValue[element].erase(superword);
        }
        shuffleMatches.erase(superword);
      }
  );
}

LogicalResult ShuffleTwoSuperwords::match(Superword* superword) {
  if (superword->constant()) {
    return failure();
  }
  SmallPtrSet<Superword*, 32> relevantSuperwords;
  for (unsigned lane = 0; lane < superword->numLanes(); ++lane) {
    if (superword->hasAlteredSemanticsInLane(lane)) {
      return failure();
    }
    auto const& existingSuperwords = superwordsByValue.lookup(superword->getElement(lane));
    if (existingSuperwords.empty()) {
      return failure();
    }
    relevantSuperwords.insert(std::begin(existingSuperwords), std::end(existingSuperwords));
  }
  // Check if any pair of relevant superwords can be combined to convert the target superword.
  SmallPtrSet<Value, 4> remainingValuesFirst;
  SmallPtrSet<Value, 4> remainingValuesSecond;
  for (auto* v1 : relevantSuperwords) {
    remainingValuesFirst.insert(superword->begin(), superword->end());
    for (auto value : *v1) {
      remainingValuesFirst.erase(value);
    }
    for (auto* v2 : relevantSuperwords) {
      remainingValuesSecond = remainingValuesFirst;
      for (auto value : *v2) {
        remainingValuesSecond.erase(value);
      }
      if (!remainingValuesSecond.empty()) {
        continue;
      }
      SmallVector<int64_t, 4> indices;
      for (auto value : *superword) {
        for (size_t i = 0; i < v1->numLanes() + v2->numLanes(); ++i) {
          Value element = i < v1->numLanes() ? v1->getElement(i) : v2->getElement(i - v1->numLanes());
          if (element == value) {
            indices.emplace_back(i);
            break;
          }
        }
      }
      assert(indices.size() == superword->numLanes() && "invalid shuffle match determined");
      shuffleMatches.try_emplace(superword, v1, v2, indices);
      return success();
    }
  }
  return failure();
}

Value ShuffleTwoSuperwords::rewrite(Superword* superword, RewriterBase& rewriter) {
  assert(shuffleMatches.count(superword) && "no match determined yet for superword");
  auto const& shuffleMatch = shuffleMatches.lookup(superword);
  auto v1 = conversionManager.getValue(std::get<0>(shuffleMatch));
  auto v2 = conversionManager.getValue(std::get<1>(shuffleMatch));
  auto maskElements = std::get<2>(shuffleMatch);
  // Erase match to keep the map as small as possible.
  shuffleMatches.erase(superword);
  return rewriter.create<vector::ShuffleOp>(superword->getLoc(), v1, v2, maskElements);
}

void ShuffleTwoSuperwords::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}

// Helper functions in anonymous namespace.
namespace {

  template<typename AttributeIterator>
  DenseElementsAttr denseElements(AttributeIterator begin, AttributeIterator end, VectorType const& vectorType) {
    if (auto floatType = vectorType.getElementType().template dyn_cast<FloatType>()) {
      if (floatType.getWidth() == 32) {
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
    } else if (auto indexType = vectorType.getElementType().template dyn_cast<IndexType>()) {
      if (indexType.isSignlessIntOrIndex()) {
        SmallVector<int64_t, 4> array;
        while (begin != end) {
          array.push_back(begin->template cast<IntegerAttr>().getInt());
          ++begin;
        }
        return DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<int64_t>>(array));
      } else if (indexType.isUnsignedInteger()) {
        SmallVector<uint64_t, 4> array;
        while (begin != end) {
          array.push_back(begin->template cast<IntegerAttr>().getUInt());
          ++begin;
        }
        return DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<uint64_t>>(array));
      }
      SmallVector<int64_t, 4> array;
      while (begin != end) {
        array.push_back(begin->template cast<IntegerAttr>().getSInt());
        ++begin;
      }
      return DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<int64_t>>(array));
    } else if (auto intType = vectorType.getElementType().template dyn_cast<IntegerType>()) {
      if (intType.isSignlessIntOrIndex()) {
        SmallVector<int64_t, 4> array;
        while (begin != end) {
          array.push_back(begin->template cast<IntegerAttr>().getInt());
          ++begin;
        }
        return DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<int64_t>>(array));
      } else if (intType.isUnsignedInteger()) {
        SmallVector<uint64_t, 4> array;
        while (begin != end) {
          array.push_back(begin->template cast<IntegerAttr>().getUInt());
          ++begin;
        }
        return DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<uint64_t>>(array));
      }
      SmallVector<int64_t, 4> array;
      while (begin != end) {
        array.push_back(begin->template cast<IntegerAttr>().getSInt());
        ++begin;
      }
      return DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<int64_t>>(array));
    }
    llvm_unreachable("illegal vector element type");
  }

}

// === VectorizeConstant === //

Value VectorizeConstant::rewrite(Superword* superword, RewriterBase& rewriter) {
  SmallVector<Attribute, 4> constants;
  for (auto const& value : *superword) {
    constants.emplace_back(static_cast<ConstantOp>(value.getDefiningOp()).value());
  }
  auto const& elements = denseElements(std::begin(constants), std::end(constants), superword->getVectorType());
  return conversionManager.getOrCreateConstant(superword->getLoc(), elements);
}

void VectorizeConstant::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}

// === VectorizeSPNConstant === //

Value VectorizeSPNConstant::rewrite(Superword* superword, RewriterBase& rewriter) {
  DenseElementsAttr elements;
  if (auto logType = superword->getElementType().dyn_cast<LogType>()) {
    if (logType.getBaseType().getIntOrFloatBitWidth() == 32) {
      SmallVector<float, 4> array;
      for (auto const& element : *superword) {
        array.push_back((float) static_cast<SPNConstant>(element.getDefiningOp()).valueAttr().getValueAsDouble());
      }
      elements = DenseElementsAttr::get(superword->getVectorType(), static_cast<llvm::ArrayRef<float>>(array));
    } else {
      SmallVector<double, 4> array;
      for (auto const& element : *superword) {
        array.push_back(static_cast<SPNConstant>(element.getDefiningOp()).valueAttr().getValueAsDouble());
      }
      elements = DenseElementsAttr::get(superword->getVectorType(), static_cast<llvm::ArrayRef<double>>(array));
    }
  } else {
    SmallVector<Attribute, 4> constants;
    for (auto const& value : *superword) {
      constants.emplace_back(static_cast<ConstantOp>(value.getDefiningOp()).valueAttr());
    }
    elements = denseElements(std::begin(constants), std::end(constants), superword->getVectorType());
  }
  return conversionManager.getOrCreateConstant(superword->getLoc(), elements);
}

void VectorizeSPNConstant::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}

// === CreateConsecutiveLoad === //

LogicalResult CreateConsecutiveLoad::match(Superword* superword) {
  if (!consecutiveLoads(superword->begin(), superword->end())) {
    // Pattern only applicable to consecutive loads.
    return failure();
  }
  return success();
}

Value CreateConsecutiveLoad::rewrite(Superword* superword, RewriterBase& rewriter) {
  auto batchRead = cast<SPNBatchRead>(superword->getElement(0).getDefiningOp());
  auto sampleIndex =
      conversionManager.getOrCreateConstant(superword->getLoc(), rewriter.getIndexAttr(batchRead.sampleIndex()));
  ValueRange indices{batchRead.batchIndex(), sampleIndex};
  return rewriter.create<vector::LoadOp>(superword->getLoc(),
                                         superword->getVectorType(),
                                         batchRead.batchMem(),
                                         indices);
}

void CreateConsecutiveLoad::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}

// === CreateGatherLoad === //

LogicalResult CreateGatherLoad::match(Superword* superword) {
  Value batchMem = nullptr;
  Value batchIndex = nullptr;
  for (auto element : *superword) {
    auto batchRead = element.getDefiningOp<SPNBatchRead>();
    if (!batchRead) {
      return failure();
    }
    // We can only gather from the same memory location.
    if (!batchMem) {
      batchMem = batchRead.batchMem();
    } else if (batchRead.batchMem() != batchMem) {
      return failure();
    }
    if (!batchIndex) {
      batchIndex = batchRead.batchIndex();
      // We require the batch index to be 0.
      if (auto* definingOp = batchIndex.getDefiningOp()) {
        auto constant = dyn_cast<ConstantOp>(definingOp);
        if (!constant || !constant.getType().isIntOrIndex() || constant.getValue().cast<IntegerAttr>().getInt() != 0) {
          return failure();
        }
      } else {
        return failure();
      }
    } else if (batchRead.batchIndex() != batchIndex) {
      return failure();
    }
  }
  return success();
}

// Helper function in anonymous namespace.
namespace {
  template<typename T>
  DenseElementsAttr constantPassThrough(VectorType const& vectorType) {
    SmallVector<T, 4> elements;
    for (auto i = 0; i < vectorType.getNumElements(); ++i) {
      elements.template emplace_back(T());
    }
    return DenseElementsAttr::get(vectorType, static_cast<ArrayRef<T>>(elements));
  }
}

Value CreateGatherLoad::rewrite(Superword* superword, RewriterBase& rewriter) {
  Value base = nullptr;
  Value index = nullptr;
  SmallVector<uint32_t, 4> samples;
  SmallVector<bool, 4> maskBits;
  for (auto element : *superword) {
    auto batchRead = cast<SPNBatchRead>(element.getDefiningOp());
    if (!base && !index) {
      base = batchRead.batchMem();
      index = batchRead.batchIndex();
    }
    samples.emplace_back(batchRead.sampleIndex());
    maskBits.emplace_back(true);
  }

  // Access the base memref beginning at [0, 0].
  SmallVector<Value, 2> indices{index, index};

  auto loc = superword->getLoc();
  auto vectorType = superword->getVectorType();

  auto indexType = VectorType::get(vectorType.getShape(), rewriter.getI32Type());
  auto indexElements = DenseElementsAttr::get(indexType, static_cast<ArrayRef<uint32_t>>(samples));
  auto indexVector = conversionManager.getOrCreateConstant(loc, indexElements);

  auto maskType = VectorType::get(vectorType.getShape(), rewriter.getI1Type());
  auto maskElements = DenseElementsAttr::get(maskType, static_cast<ArrayRef<bool>>(maskBits));
  auto mask = conversionManager.getOrCreateConstant(loc, maskElements);

  DenseElementsAttr passThroughElements;
  if (superword->getElementType().isIntOrIndex()) {
    passThroughElements = constantPassThrough<int>(vectorType);
  } else if (superword->getElementType().isF32()) {
    passThroughElements = constantPassThrough<float>(vectorType);
  } else if (superword->getElementType().isF64()) {
    passThroughElements = constantPassThrough<double>(vectorType);
  } else {
    llvm_unreachable("unsupported vector element type for gather op");
  }
  auto passThrough = conversionManager.getOrCreateConstant(loc, passThroughElements);

  return rewriter.create<vector::GatherOp>(loc, vectorType, base, indices, indexVector, mask, passThrough);
}

void CreateGatherLoad::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}

// === VectorizeAdd === //

Value VectorizeAdd::rewrite(Superword* superword, RewriterBase& rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  return rewriter.create<AddFOp>(superword->getLoc(), superword->getVectorType(), operands);
}

void VectorizeAdd::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}

// === VectorizeMul === //

Value VectorizeMul::rewrite(Superword* superword, RewriterBase& rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  return rewriter.create<MulFOp>(superword->getLoc(), superword->getVectorType(), operands);
}

void VectorizeMul::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}

// === VectorizeGaussian === //

Value VectorizeGaussian::rewrite(Superword* superword, RewriterBase& rewriter) {

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
      meansArray.emplace_back((float) static_cast<SPNGaussianLeaf>(value.getDefiningOp()).meanAttr().getValueAsDouble());
      float stddev = (float) static_cast<SPNGaussianLeaf>(value.getDefiningOp()).stddevAttr().getValueAsDouble();
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
      meansArray.emplace_back(static_cast<SPNGaussianLeaf>(value.getDefiningOp()).meanAttr().getValueAsDouble());
      double stddev = static_cast<SPNGaussianLeaf>(value.getDefiningOp()).stddevAttr().getValueAsDouble();
      variancesArray.emplace_back(-2 * stddev * stddev);
      rootsArray.emplace_back(std::sqrt(2.0 * M_PI * stddev * stddev));
    }
    means = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(meansArray));
    variances = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(variancesArray));
    roots = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(rootsArray));
  }

  // (x - mean)
  auto inputVector = conversionManager.getValue(superword->getOperand(0));
  inputVector = util::extendTruncateOrGetVector(inputVector, vectorType, rewriter);
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
  return rewriter.create<DivFOp>(superword->getLoc(), gaussianVector, rootsVector);
}

void VectorizeGaussian::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}

// === VectorizeLogAdd === //

// Helper functions in anonymous namespace.
namespace {
  template<typename T>
  DenseElementsAttr denseConstant(T constant, VectorType vectorType) {
    SmallVector<T, 4> array(vectorType.getNumElements());
    for (auto i = 0; i < vectorType.getNumElements(); ++i) {
      array[i] = constant;
    }
    return DenseElementsAttr::get(vectorType, static_cast<ArrayRef<T>>(array));
  }
}

Value VectorizeLogAdd::rewrite(Superword* superword, RewriterBase& rewriter) {
  // Rewrite 'ln(x + y)' with 'a = ln(x), b = ln(y) and a > b' as 'a + ln(e^(b - a) + 1)'.
  auto lhs = conversionManager.getValue(superword->getOperand(0));
  auto rhs = conversionManager.getValue(superword->getOperand(1));
  auto compare = rewriter.create<CmpFOp>(superword->getLoc(), CmpFPredicate::OGT, lhs, rhs);
  auto a = rewriter.create<SelectOp>(superword->getLoc(), compare, lhs, rhs);
  auto b = rewriter.create<SelectOp>(superword->getLoc(), compare, rhs, lhs);
  Value vectorOp = rewriter.create<SubFOp>(superword->getLoc(), b, a);
  vectorOp = rewriter.create<math::ExpOp>(superword->getLoc(), vectorOp);
  vectorOp = rewriter.create<math::Log1pOp>(superword->getLoc(), vectorOp);
  return rewriter.create<AddFOp>(superword->getLoc(), a, vectorOp);
}

void VectorizeLogAdd::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}

// === VectorizeLogMul === //

Value VectorizeLogMul::rewrite(Superword* superword, RewriterBase& rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  return rewriter.create<AddFOp>(superword->getLoc(), superword->getVectorType(), operands);
}

void VectorizeLogMul::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}

// === VectorizeLogGaussian === //

Value VectorizeLogGaussian::rewrite(Superword* superword, RewriterBase& rewriter) {

  auto vectorType = superword->getVectorType();

  // Calculate log-space Gaussian distribution using -ln(stddev) - 0.5 * ln(2*pi) - (x - mean)^2 / (2 * stddev^2)
  DenseElementsAttr means;
  DenseElementsAttr minuends;
  DenseElementsAttr factors;
  if (vectorType.getElementType().cast<FloatType>().getWidth() == 32) {
    SmallVector<float, 4> meansArray;
    SmallVector<float, 4> minuendsArray;
    SmallVector<float, 4> factorsArray;
    for (auto const& value : *superword) {
      meansArray.emplace_back((float) static_cast<SPNGaussianLeaf>(value.getDefiningOp()).meanAttr().getValueAsDouble());
      float stddev = (float) static_cast<SPNGaussianLeaf>(value.getDefiningOp()).stddevAttr().getValueAsDouble();
      minuendsArray.emplace_back(-logf(stddev) - 0.5 * logf(2 * M_PIf32));
      factorsArray.emplace_back(1.0 / (2 * stddev * stddev));
    }
    means = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<float>>(meansArray));
    minuends = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<float>>(minuendsArray));
    factors = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<float>>(factorsArray));
  } else {
    SmallVector<double, 4> meansArray;
    SmallVector<double, 4> minuendsArray;
    SmallVector<double, 4> factorsArray;
    for (auto const& value : *superword) {
      meansArray.emplace_back(static_cast<SPNGaussianLeaf>(value.getDefiningOp()).meanAttr().getValueAsDouble());
      double stddev = static_cast<SPNGaussianLeaf>(value.getDefiningOp()).stddevAttr().getValueAsDouble();
      minuendsArray.emplace_back(-log(stddev) - 0.5 * log(2 * M_PI));
      factorsArray.emplace_back(1.0 / (2 * stddev * stddev));
    }
    means = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(meansArray));
    minuends = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(minuendsArray));
    factors = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(factorsArray));
  }

  // x - mean
  auto inputVector = conversionManager.getValue(superword->getOperand(0));
  inputVector = util::extendTruncateOrGetVector(inputVector, vectorType, rewriter);
  auto meanVector = conversionManager.getOrCreateConstant(superword->getLoc(), means);
  Value gaussianVector = rewriter.create<SubFOp>(superword->getLoc(), vectorType, inputVector, meanVector);

  // (x - mean)^2
  gaussianVector = rewriter.create<MulFOp>(superword->getLoc(), vectorType, gaussianVector, gaussianVector);

  // (x - mean)^2 / (2 * stddev^2)
  auto factorsVector = conversionManager.getOrCreateConstant(superword->getLoc(), factors);
  gaussianVector = rewriter.create<MulFOp>(superword->getLoc(), vectorType, gaussianVector, factorsVector);

  // -ln(stddev) - 0.5 * ln(2*pi) - (x - mean)^2 / (2 * stddev^2)
  auto minuendVector = conversionManager.getOrCreateConstant(superword->getLoc(), minuends);
  return rewriter.create<SubFOp>(superword->getLoc(), vectorType, minuendVector, gaussianVector);
}

void VectorizeLogGaussian::accept(PatternVisitor& visitor, Superword* superword) const {
  visitor.visit(this, superword);
}
