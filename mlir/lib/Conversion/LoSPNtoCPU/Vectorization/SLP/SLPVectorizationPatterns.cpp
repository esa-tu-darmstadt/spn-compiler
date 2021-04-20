//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

// Helper functions in anonymous namespace.
namespace {

  template<typename ValueIterator>
  Value broadcastFirstInsertRest(ValueIterator begin,
                                 ValueIterator end,
                                 VectorType const& vectorType,
                                 PatternRewriter& rewriter) {
    Value vectorOp = rewriter.create<vector::BroadcastOp>(begin->getLoc(), vectorType, *begin);
    unsigned position = 1;
    while (++begin != end) {
      vectorOp = rewriter.create<vector::InsertElementOp>(begin->getLoc(), *begin, vectorOp, position++);
    }
    return vectorOp;
  }

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

  llvm::SmallVector<SLPNode*> postOrder(SLPNode* root) {
    llvm::SmallVector<SLPNode*> order;
    for (auto* operand : root->getOperands()) {
      assert(std::find(std::begin(order), std::end(order), operand) == std::end(order));
      order.append(postOrder(operand));
    }
    order.emplace_back(root);
    return order;
  }

  DenseMap<NodeVector*, SmallVector<size_t, 4>> getEscapingLanesMap(SLPNode* root) {
    DenseMap<NodeVector*, DenseMap<mlir::Value, unsigned>> outsideUses;
    for (auto* node : postOrder(root)) {
      for (size_t i = 0; i < node->numVectors(); ++i) {
        auto* vector = node->getVector(i);
        for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
          auto const& element = vector->getElement(lane);
          // Skip duplicate (splat) values.
          if (outsideUses[vector].count(element)) {
            continue;
          }
          outsideUses[vector][element] = std::distance(std::begin(element.getUses()), std::end(element.getUses()));
          for (size_t j = 0; j < vector->numOperands(); ++j) {
            auto* operand = vector->getOperand(j);
            assert(outsideUses[operand][operand->getElement(lane)] > 0);
            outsideUses[operand][operand->getElement(lane)]--;
          }
        }
      }
    }

    DenseMap<NodeVector*, SmallVector<size_t, 4>> escapingLanes;
    for (auto const& entry : outsideUses) {
      auto* vector = entry.first;
      for (auto const& useAndCount : entry.second) {
        auto const& value = useAndCount.first;
        auto const& count = useAndCount.second;
        if (count > 0) {
          for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
            if (vector->getElement(lane) == value) {
              escapingLanes[vector].emplace_back(lane);
              break;
            }
          }
        }
      }
    }
    return escapingLanes;
  }

  bool containsBlockArgs(NodeVector* vector) {
    return std::any_of(vector->begin(), vector->end(), [&](Value const& value) {
      return value.isa<BlockArgument>();
    });
  }

  template<typename ValueIterator>
  bool valuesVectorizable(ValueIterator begin, ValueIterator end) {
    return std::all_of(begin, end, [&](Value const& value) {
      return vectorizable(value);
    });
  }

}

SLPVectorPatternRewriter::SLPVectorPatternRewriter(mlir::MLIRContext* ctx) : PatternRewriter{ctx} {}

LogicalResult SLPVectorPatternRewriter::rewrite(SLPNode* root) {

  // The current vector being transformed.
  NodeVector* vector = nullptr;

  // Maps individual SLP node vectors to finished vector operations.
  DenseMap<NodeVector*, Operation*> vectorizedOps;

  // Marks operations as potentially erasable (potentially because they might have escaping uses).
  // We delete them *after* SLP graph conversion to avoid running into NULL operands during conversion.
  SmallPtrSet<NodeVector*, 32> erasableOps;

  OwningRewritePatternList patterns;
  populateSLPVectorizationPatterns(patterns, context, vector, vectorizedOps, erasableOps);
  FrozenRewritePatternList frozenPatterns(std::move(patterns));

  PatternApplicator applicator{frozenPatterns};
  applicator.applyDefaultCostModel();

  // Stores escaping uses for vector extractions that might be necessary later on.
  auto const& escapingLanes = getEscapingLanesMap(root);

  // No need to extract/remove values more than once in case they appear more than once (e.g. in splat mode).
  SmallPtrSet<Value, 4> finishedValues;

  // Traverse the SLP graph in postorder and apply the vectorization patterns.
  for (auto* node : postOrder(root)) {

    // Also traverse nodes in postorder to properly handle multinodes.
    for (size_t vectorIndex = node->numVectors(); vectorIndex-- > 0;) {
      vector = node->getVector(vectorIndex);
      // Rewrite vector by applying any matching pattern.
      if (!valuesVectorizable(vector->begin(), vector->end())) {
        continue;
      }
      if (containsBlockArgs(vector)) {
        auto const& elementType = vector->getElement(0).getType();
        auto const& type = VectorType::get(static_cast<unsigned>(vector->numLanes()), elementType);
        setInsertionPoint(firstUser(vector->begin(), vector->end()));
        auto vectorOp = broadcastFirstInsertRest(vector->begin(), vector->end(), type, *this);
        vectorizedOps[vector] = vectorOp.getDefiningOp();
      } else {
        auto* vectorOp = vector->begin()->getDefiningOp();
        if (failed(applicator.matchAndRewrite(vectorOp, *this))) {
          vectorOp->emitOpError("SLP pattern application failed (did you forget to specify the pattern?)");
        }
      }
    }

    // Create vector extractions for escaping uses & erase superfluous operations.
    for (size_t vectorIndex = node->numVectors(); vectorIndex-- > 0;) {
      vector = node->getVector(vectorIndex);
      if (vectorizedOps.count(vector) && erasableOps.count(vector)) {
        for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
          auto const& vectorValue = vector->getElement(lane);
          if (finishedValues.contains(vectorValue)) {
            continue;
          }
          auto const& lanesToExtract = escapingLanes.lookup(vector);
          if (std::find(std::begin(lanesToExtract), std::end(lanesToExtract), lane) != std::end(lanesToExtract)) {
            auto const& source = vectorizedOps[vector]->getResult(0);
            setInsertionPoint(firstOccurrence(std::begin(vectorValue.getUsers()), std::end(vectorValue.getUsers())));
            auto extractOp = create<vector::ExtractElementOp>(vectorValue.getLoc(), source, lane);
            vectorValue.replaceAllUsesWith(extractOp.result());
          } else {
            vectorValue.dropAllUses();
          }
          finishedValues.insert(vectorValue);
          eraseOp(vectorValue.getDefiningOp());
        }
      }
    }
  }

  return success();
}

LogicalResult VectorizeConstant::matchAndRewrite(ConstantOp constantOp, PatternRewriter& rewriter) const {

  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), constantOp.getType());

  rewriter.setInsertionPoint(firstUser(vector->begin(), vector->end()));

  if (!vector->isUniform()) {
    auto vectorVal = broadcastFirstInsertRest(vector->begin(), vector->end(), vectorType, rewriter);
    vectorizedOps[vector] = vectorVal.getDefiningOp();
    return success();
  }

  SmallVector<Attribute, 4> constants;
  for (auto const& value : *vector) {
    constants.emplace_back(value.getDefiningOp<ConstantOp>().value());
  }
  auto const& elements = denseFloatingPoints(std::begin(constants), std::end(constants), vectorType);

  auto constVector = rewriter.create<mlir::ConstantOp>(constantOp->getLoc(), elements);

  vectorizedOps[vector] = constVector;
  erasableOps.insert(vector);

  return success();
}

LogicalResult VectorizeBatchRead::matchAndRewrite(SPNBatchRead batchReadOp, PatternRewriter& rewriter) const {

  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), batchReadOp.getType());

  rewriter.setInsertionPoint(firstUser(vector->begin(), vector->end()));

  if (!vector->isUniform() || !consecutiveLoads(vector->begin(), vector->end())) {
    auto vectorVal = broadcastFirstInsertRest(vector->begin(), vector->end(), vectorType, rewriter);
    vectorizedOps[vector] = vectorVal.getDefiningOp();
  } else {

    auto batchReadLoc = batchReadOp->getLoc();
    auto memIndex = rewriter.create<ConstantOp>(batchReadLoc, rewriter.getIndexAttr(batchReadOp.sampleIndex()));
    ValueRange indices{batchReadOp.batchIndex(), memIndex};
    auto vectorLoad = rewriter.create<vector::LoadOp>(batchReadLoc, vectorType, batchReadOp.batchMem(), indices);

    vectorizedOps[vector] = vectorLoad;
    erasableOps.insert(vector);
  }

  return success();
}

LogicalResult VectorizeAdd::matchAndRewrite(SPNAdd addOp, PatternRewriter& rewriter) const {

  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), addOp.getType());

  rewriter.setInsertionPoint(firstUser(vector->begin(), vector->end()));

  if (!vector->isUniform()) {
    auto vectorVal = broadcastFirstInsertRest(vector->begin(), vector->end(), vectorType, rewriter);
    vectorizedOps[vector] = vectorVal.getDefiningOp();
    return success();
  }

  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < addOp.getNumOperands(); ++i) {
    auto* operandOp = vectorizedOps[vector->getOperand(i)];
    if (!operandOp) {
      return rewriter.notifyMatchFailure(addOp, "operation's operands have not yet been (fully) vectorized");
    }
    operands.emplace_back(operandOp->getResult(0));
  }

  auto vectorAddOp = rewriter.create<AddFOp>(addOp->getLoc(), vectorType, operands);

  vectorizedOps[vector] = vectorAddOp;
  erasableOps.insert(vector);

  return success();
}

LogicalResult VectorizeMul::matchAndRewrite(SPNMul mulOp, PatternRewriter& rewriter) const {

  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), mulOp.getType());

  rewriter.setInsertionPoint(firstUser(vector->begin(), vector->end()));

  if (!vector->isUniform()) {
    auto vectorVal = broadcastFirstInsertRest(vector->begin(), vector->end(), vectorType, rewriter);
    vectorizedOps[vector] = vectorVal.getDefiningOp();
    return success();
  }

  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < mulOp.getNumOperands(); ++i) {
    auto* operandOp = vectorizedOps[vector->getOperand(i)];
    if (!operandOp) {
      return rewriter.notifyMatchFailure(mulOp, "operation's operands have not yet been (fully) vectorized");
    }
    operands.emplace_back(operandOp->getResult(0));
  }

  auto vectorAddOp = rewriter.create<MulFOp>(mulOp->getLoc(), vectorType, operands);

  vectorizedOps[vector] = vectorAddOp;
  erasableOps.insert(vector);

  return success();
}

LogicalResult VectorizeGaussian::matchAndRewrite(SPNGaussianLeaf gaussianOp, PatternRewriter& rewriter) const {

  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), gaussianOp.getType());
  auto const& firstValue = firstOccurrence(vector->begin(), vector->end());

  if (firstValue.isa<BlockArgument>()) {
    rewriter.setInsertionPointAfterValue(firstValue);
  } else {
    rewriter.setInsertionPoint(firstValue.getDefiningOp());
  }

  if (!vector->isUniform()) {
    auto vectorVal = broadcastFirstInsertRest(vector->begin(), vector->end(), vectorType, rewriter);
    vectorizedOps[vector] = vectorVal.getDefiningOp();
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
  auto* inputOp = vectorizedOps[vector->getOperand(0)];
  if (!inputOp) {
    return rewriter.notifyMatchFailure(gaussianOp, "operation's operands have not yet been (fully) vectorized");
  }
  Value inputVector = inputOp->getResult(0);

  // Calculate Gaussian distribution using e^(-0.5 * ((x - mean) / stddev)^2)) / (stddev * sqrt(2 * PI))
  auto const& gaussianLoc = gaussianOp->getLoc();

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
  gaussianVector = rewriter.create<math::ExpOp>(gaussianLoc, gaussianVector);

  // e^(-0.5 * ((x - mean) / stddev)^2)) / (stddev * sqrt(2 * PI))
  auto coefficientVector = rewriter.create<ConstantOp>(gaussianLoc, coefficients);
  gaussianVector = rewriter.create<MulFOp>(gaussianLoc, coefficientVector, gaussianVector);

  vectorizedOps[vector] = gaussianVector.getDefiningOp();
  erasableOps.insert(vector);

  return success();
}
