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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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

  /// Strip the log space property off an operation if present, otherwise do nothing.
  Value stripLogOrValue(Value value, RewriterBase& rewriter) {
    if (auto logType = value.getType().dyn_cast<LogType>()) {
      return rewriter.create<SPNStripLog>(value.getLoc(), value, logType.getBaseType());
    }
    return value;
  }

  /// Might be useful in the future.
  // NOLINTNEXTLINE(clang-diagnostic-unused-function)
  [[maybe_unused]] Value castToFloatOrValue(Value value, FloatType targetType, RewriterBase& rewriter) {
    if (auto floatType = value.getType().dyn_cast<FloatType>()) {
      if (floatType.getWidth() < targetType.getWidth()) {
        return rewriter.create<LLVM::FPExtOp>(value.getLoc(), targetType, value);
      } else if (floatType.getWidth() > targetType.getWidth()) {
        return rewriter.create<LLVM::FPTruncOp>(value.getLoc(), targetType, value);
      } else {
        return value;
      }
    } else if (auto intType = value.getType().dyn_cast<IntegerType>()) {
      if (intType.isSigned()) {
        return rewriter.create<arith::SIToFPOp>(value.getLoc(), targetType, value);
      }
      return rewriter.create<arith::UIToFPOp>(value.getLoc(), targetType, value);
    } else if (value.getType().isa<IndexType>()) {
      auto valueAsInt = rewriter.create<arith::IndexCastOp>(value.getLoc(), rewriter.getI64Type(), value);
      return rewriter.create<arith::UIToFPOp>(value.getLoc(), targetType, valueAsInt);
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
      // TODO: Check if the new parameters make sense.
      if (!OperationEquivalence::isEquivalentTo(definingOp, firstOp, nullptr, nullptr, OperationEquivalence::Flags::None)) {
        return failure();
      }
    } else if (firstOp || superword->getElement(lane) != superword->getElement(0)) {
      return failure();
    }
  }
  return success();
}

Value BroadcastSuperword::rewrite(Superword* superword, RewriterBase& rewriter) {
  auto element = stripLogOrValue(superword->getElement(0), rewriter);
  return rewriter.create<vector::BroadcastOp>(element.getLoc(), superword->getVectorType(), element);
}

void BroadcastSuperword::accept(PatternVisitor& visitor, Superword const* superword) const {
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
  for (auto element: *superword) {
    if (++elementCounts[element] > maxCount) {
      broadcastVal = element;
      maxCount = elementCounts[element];
    }
  }
  assert(broadcastVal);
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

void BroadcastInsertSuperword::accept(PatternVisitor& visitor, Superword const* superword) const {
  visitor.visit(this, superword);
}

// === ShuffleTwoSuperwords === //

ShuffleTwoSuperwords::ShuffleTwoSuperwords(ConversionManager& conversionManager) : SLPVectorizationPattern(
    conversionManager) {
  // The shuffle pattern needs to know which superwords are available for shuffling.
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
          superwordsByValue[element].remove(superword);
        }
        shuffleMatches.erase(superword);
      }
  );
}

LogicalResult ShuffleTwoSuperwords::match(Superword* superword) {
  if (superword->constant()) {
    return failure();
  }
  // For determinism purposes, use a set vector with deterministic iteration order instead of a set.
  llvm::SmallSetVector<Superword*, 32> relevantSuperwords;
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
  for (auto* v1 : relevantSuperwords) {
    SmallPtrSet<Value, 4> remainingValuesFirst(superword->begin(), superword->end());
    for (size_t lane = 0; lane < v1->numLanes(); ++lane) {
      if (v1->hasAlteredSemanticsInLane(lane)) {
        continue;
      }
      remainingValuesFirst.erase(v1->getElement(lane));
    }
    for (auto* v2 : relevantSuperwords) {
      auto remainingValuesSecond = remainingValuesFirst;
      for (size_t lane = 0; lane < v2->numLanes(); ++lane) {
        if (v2->hasAlteredSemanticsInLane(lane)) {
          continue;
        }
        remainingValuesSecond.erase(v2->getElement(lane));
      }
      if (!remainingValuesSecond.empty()) {
        continue;
      }
      SmallVector<int64_t, 4> indices;
      for (auto value : *superword) {
        for (size_t index = 0; index < v1->numLanes() + v2->numLanes(); ++index) {
          auto* v = index < v1->numLanes() ? v1 : v2;
          size_t lane = index < v1->numLanes() ? index : index - v1->numLanes();
          if (v->getElement(lane) == value) {
            if (v->hasAlteredSemanticsInLane(lane)) {
              continue;
            }
            indices.emplace_back(index);
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

void ShuffleTwoSuperwords::accept(PatternVisitor& visitor, Superword const* superword) const {
  visitor.visit(this, superword);
}

// Helper functions in anonymous namespace.
namespace {

  /// Creates a constant of type vectorType consisting of all attributes in [begin, end).
  template<typename AttributeIterator>
  DenseElementsAttr denseElements(AttributeIterator begin, AttributeIterator end, VectorType const& vectorType) {
    if (auto floatType = vectorType.getElementType().template dyn_cast<FloatType>()) {
      if (floatType.getWidth() == 32) {
        SmallVector<float, 4> array;
        while (begin != end) {
          array.push_back(static_cast<float>(begin->template cast<FloatAttr>().getValue().convertToDouble()));
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
  for (auto value : *superword) {
    // ConstantOp.
    if (auto op = value.getDefiningOp<arith::ConstantOp>()) {
      constants.emplace_back(op.getValue());
    }
      // SPNConstant op.
    else {
      constants.emplace_back(static_cast<SPNConstant>(value.getDefiningOp()).getValueAttr());
    }
  }
  auto const& elements = denseElements(std::begin(constants), std::end(constants), superword->getVectorType());
  // TODO: This is guesswork. I think vector types were moved form builtin into their own dialect. That's
  // we need that dialect with the implemented materializeConstants(). Additionally, we need the type information.
  Dialect *dialect = rewriter.getContext()->getLoadedDialect<vector::VectorDialect>();
  return conversionManager.getOrCreateConstant(superword->getLoc(), elements, elements.getType(), dialect);
}

void VectorizeConstant::accept(PatternVisitor& visitor, Superword const* superword) const {
  visitor.visit(this, superword);
}

// === CreateConsecutiveLoad === //

LogicalResult CreateConsecutiveLoad::match(Superword* superword) {
  // Pattern only applicable to consecutive loads.
  return success(consecutiveLoads(superword->begin(), superword->end()));
}

Value CreateConsecutiveLoad::rewrite(Superword* superword, RewriterBase& rewriter) {
  auto batchRead = cast<SPNBatchRead>(superword->getElement(0).getDefiningOp());
  ValueRange indices{
      batchRead.getDynamicIndex(),
      conversionManager.getOrCreateConstant(superword->getLoc(), rewriter.getIndexAttr(batchRead.getStaticIndex()))
  };
  return rewriter.create<vector::LoadOp>(superword->getLoc(),
                                         superword->getVectorType(),
                                         batchRead.getBatchMem(),
                                         indices);
}

void CreateConsecutiveLoad::accept(PatternVisitor& visitor, Superword const* superword) const {
  visitor.visit(this, superword);
}

// === CreateGatherLoad === //

LogicalResult CreateGatherLoad::match(Superword* superword) {
  Value batchMem = nullptr;
  Value dynamicIndex = nullptr;
  for (auto element : *superword) {
    auto batchRead = element.getDefiningOp<SPNBatchRead>();
    if (!batchRead) {
      return failure();
    }
    // We can only gather from the same memory location.
    if (!batchMem) {
      batchMem = batchRead.getBatchMem();
    } else if (batchRead.getBatchMem() != batchMem) {
      return failure();
    }
    if (!dynamicIndex) {
      dynamicIndex = batchRead.getDynamicIndex();
      // We require the dynamic index to be 0.
      if (auto* definingOp = dynamicIndex.getDefiningOp()) {
        auto constant = dyn_cast<arith::ConstantOp>(definingOp);
        if (!constant || !constant.getType().isIntOrIndex() || constant.getValue().cast<IntegerAttr>().getInt() != 0) {
          return failure();
        }
      } else {
        return failure();
      }
    } else if (batchRead.getDynamicIndex() != dynamicIndex) {
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
      base = batchRead.getBatchMem();
      index = batchRead.getDynamicIndex();
    }
    samples.emplace_back(batchRead.getStaticIndex());
    maskBits.emplace_back(true);
  }

  // Access the base memref beginning at [0, 0].
  SmallVector<Value, 2> indices{index, index};

  auto loc = superword->getLoc();
  auto vectorType = superword->getVectorType();

  Dialect *vectorDialect = rewriter.getContext()->getLoadedDialect<vector::VectorDialect>();

  auto indexType = VectorType::get(vectorType.getShape(), rewriter.getI32Type());
  auto indexElements = DenseElementsAttr::get(indexType, static_cast<ArrayRef<uint32_t>>(samples));
  auto indexVector = conversionManager.getOrCreateConstant(loc, indexElements, indexElements.getType(), vectorDialect);

  auto maskType = VectorType::get(vectorType.getShape(), rewriter.getI1Type());
  auto maskElements = DenseElementsAttr::get(maskType, static_cast<ArrayRef<bool>>(maskBits));
  auto mask = conversionManager.getOrCreateConstant(loc, maskElements, maskElements.getType(), vectorDialect);

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
  auto passThrough = conversionManager.getOrCreateConstant(loc, passThroughElements, passThroughElements.getType(), vectorDialect);

  return rewriter.create<vector::GatherOp>(loc, vectorType, base, indices, indexVector, mask, passThrough);
}

void CreateGatherLoad::accept(PatternVisitor& visitor, Superword const* superword) const {
  visitor.visit(this, superword);
}

// === VectorizeAdd === //

Value VectorizeAdd::rewrite(Superword* superword, RewriterBase& rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  return rewriter.create<arith::AddFOp>(superword->getLoc(), superword->getVectorType(), operands);
}

void VectorizeAdd::accept(PatternVisitor& visitor, Superword const* superword) const {
  visitor.visit(this, superword);
}

// === VectorizeMul === //

Value VectorizeMul::rewrite(Superword* superword, RewriterBase& rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  return rewriter.create<arith::MulFOp>(superword->getLoc(), superword->getVectorType(), operands);
}

void VectorizeMul::accept(PatternVisitor& visitor, Superword const* superword) const {
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
    for (auto value : *superword) {
      meansArray.emplace_back((float) static_cast<SPNGaussianLeaf>(value.getDefiningOp()).getMeanAttr().getValueAsDouble());
      float stddev = (float) static_cast<SPNGaussianLeaf>(value.getDefiningOp()).getStddevAttr().getValueAsDouble();
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
    for (auto value : *superword) {
      meansArray.emplace_back(static_cast<SPNGaussianLeaf>(value.getDefiningOp()).getMeanAttr().getValueAsDouble());
      double stddev = static_cast<SPNGaussianLeaf>(value.getDefiningOp()).getStddevAttr().getValueAsDouble();
      variancesArray.emplace_back(-2 * stddev * stddev);
      rootsArray.emplace_back(std::sqrt(2.0 * M_PI * stddev * stddev));
    }
    means = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(meansArray));
    variances = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(variancesArray));
    roots = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(rootsArray));
  }

  // (x - mean)
  auto inputVector = conversionManager.getValue(superword->getOperand(0));
  inputVector = util::castToFloatOrGetVector(inputVector, vectorType, rewriter);
  inputVector = util::extendTruncateOrGetVector(inputVector, vectorType, rewriter);
  Dialect *vectorDialect = rewriter.getContext()->getLoadedDialect<vector::VectorDialect>();
  auto meanVector = conversionManager.getOrCreateConstant(superword->getLoc(), means, means.getType(), vectorDialect);
  Value gaussianVector = rewriter.create<arith::SubFOp>(superword->getLoc(), vectorType, inputVector, meanVector);

  // ((x - mean)^2
  gaussianVector = rewriter.create<arith::MulFOp>(superword->getLoc(), vectorType, gaussianVector, gaussianVector);

  // (x - mean)^2 / (-2 * variance)
  auto varianceVector = conversionManager.getOrCreateConstant(superword->getLoc(), variances, variances.getType(), vectorDialect);
  gaussianVector = rewriter.create<arith::DivFOp>(superword->getLoc(), vectorType, gaussianVector, varianceVector);

  // e^((x - mean)^2 / (-2 * variance))
  gaussianVector = rewriter.create<math::ExpOp>(superword->getLoc(), vectorType, gaussianVector);

  // e^((x - mean)^2 / (-2 * variance)) / sqrt(2 * PI * variance)
  auto rootsVector = conversionManager.getOrCreateConstant(superword->getLoc(), roots, roots.getType(), vectorDialect);
  gaussianVector = rewriter.create<arith::DivFOp>(superword->getLoc(), gaussianVector, rootsVector);

  if (anyGaussianMarginalized(*superword)) {
    DenseElementsAttr denseOne;
    if (vectorType.getElementType().cast<FloatType>().getWidth() == 32) {
      SmallVector<float, 4> ones(superword->numLanes(), 1.0f);
      denseOne = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<float>>(ones));
    } else {
      SmallVector<double, 4> ones(superword->numLanes(), 1.0);
      denseOne = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(ones));
    }
    auto isNan = rewriter.create<arith::CmpFOp>(superword->getLoc(), arith::CmpFPredicate::UNO, inputVector, inputVector);
    auto constOne = conversionManager.getOrCreateConstant(superword->getLoc(), denseOne, denseOne.getType(), vectorDialect);
    gaussianVector = rewriter.create<arith::SelectOp>(superword->getLoc(), isNan, constOne, gaussianVector);
  }

  return gaussianVector;
}

void VectorizeGaussian::accept(PatternVisitor& visitor, Superword const* superword) const {
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
  auto compare = rewriter.create<arith::CmpFOp>(superword->getLoc(), arith::CmpFPredicate::OGT, lhs, rhs);
  auto a = rewriter.create<arith::SelectOp>(superword->getLoc(), compare, lhs, rhs);
  auto b = rewriter.create<arith::SelectOp>(superword->getLoc(), compare, rhs, lhs);
  Value vectorOp = rewriter.create<arith::SubFOp>(superword->getLoc(), b, a);
  vectorOp = rewriter.create<math::ExpOp>(superword->getLoc(), vectorOp);
  vectorOp = rewriter.create<math::Log1pOp>(superword->getLoc(), vectorOp);
  return rewriter.create<arith::AddFOp>(superword->getLoc(), a, vectorOp);
}

void VectorizeLogAdd::accept(PatternVisitor& visitor, Superword const* superword) const {
  visitor.visit(this, superword);
}

// === VectorizeLogMul === //

Value VectorizeLogMul::rewrite(Superword* superword, RewriterBase& rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  return rewriter.create<arith::AddFOp>(superword->getLoc(), superword->getVectorType(), operands);
}

void VectorizeLogMul::accept(PatternVisitor& visitor, Superword const* superword) const {
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
    for (auto value : *superword) {
      meansArray.emplace_back((float) static_cast<SPNGaussianLeaf>(value.getDefiningOp()).getMeanAttr().getValueAsDouble());
      float stddev = (float) static_cast<SPNGaussianLeaf>(value.getDefiningOp()).getStddevAttr().getValueAsDouble();
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
    for (auto value : *superword) {
      meansArray.emplace_back(static_cast<SPNGaussianLeaf>(value.getDefiningOp()).getMeanAttr().getValueAsDouble());
      double stddev = static_cast<SPNGaussianLeaf>(value.getDefiningOp()).getStddevAttr().getValueAsDouble();
      minuendsArray.emplace_back(-log(stddev) - 0.5 * log(2 * M_PI));
      factorsArray.emplace_back(1.0 / (2 * stddev * stddev));
    }
    means = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(meansArray));
    minuends = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(minuendsArray));
    factors = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(factorsArray));
  }

  // x - mean
  auto inputVector = conversionManager.getValue(superword->getOperand(0));
  inputVector = util::castToFloatOrGetVector(inputVector, vectorType, rewriter);
  inputVector = util::extendTruncateOrGetVector(inputVector, vectorType, rewriter);
  auto meanVector = conversionManager.getOrCreateConstant(superword->getLoc(), means);
  Value gaussianVector = rewriter.create<arith::SubFOp>(superword->getLoc(), vectorType, inputVector, meanVector);

  // (x - mean)^2
  gaussianVector = rewriter.create<arith::MulFOp>(superword->getLoc(), vectorType, gaussianVector, gaussianVector);

  // (x - mean)^2 / (2 * stddev^2)
  auto factorsVector = conversionManager.getOrCreateConstant(superword->getLoc(), factors);
  gaussianVector = rewriter.create<arith::MulFOp>(superword->getLoc(), vectorType, gaussianVector, factorsVector);

  // -ln(stddev) - 0.5 * ln(2*pi) - (x - mean)^2 / (2 * stddev^2)
  auto minuendVector = conversionManager.getOrCreateConstant(superword->getLoc(), minuends);
  gaussianVector = rewriter.create<arith::SubFOp>(superword->getLoc(), vectorType, minuendVector, gaussianVector);

  if (anyGaussianMarginalized(*superword)) {
    DenseElementsAttr denseZero;
    if (vectorType.getElementType().cast<FloatType>().getWidth() == 32) {
      SmallVector<float, 4> zeros(superword->numLanes(), 0.0f);
      denseZero = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<float>>(zeros));
    } else {
      SmallVector<double, 4> zeros(superword->numLanes(), 0.0);
      denseZero = DenseElementsAttr::get(vectorType, static_cast<ArrayRef<double>>(zeros));
    }
    auto isNan = rewriter.create<arith::CmpFOp>(superword->getLoc(), arith::CmpFPredicate::UNO, inputVector, inputVector);
    auto constZero = conversionManager.getOrCreateConstant(superword->getLoc(), denseZero);
    gaussianVector = rewriter.create<arith::SelectOp>(superword->getLoc(), isNan, constZero, gaussianVector);
  }

  return gaussianVector;
}

void VectorizeLogGaussian::accept(PatternVisitor& visitor, Superword const* superword) const {
  visitor.visit(this, superword);
}
