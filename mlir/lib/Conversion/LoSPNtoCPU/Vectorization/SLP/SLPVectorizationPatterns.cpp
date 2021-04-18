//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/IR/BlockAndValueMapping.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

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

  template<typename OperationIterator>
  SmallVector<Value, 4> results(OperationIterator begin, OperationIterator end) {
    SmallVector<Value, 4> elements;
    while (begin != end) {
      elements.emplace_back((*begin)->getResult(0));
      ++begin;
    }
    return elements;
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

}

LogicalResult VectorizeConstant::matchAndRewrite(ConstantOp constantOp, PatternRewriter& rewriter) const {

  auto* node = this->parentNodes.lookup(constantOp);

  auto const& vectorIndex = node->getVectorIndex(constantOp);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), constantOp.getType());

  rewriter.setInsertionPoint(firstUser(std::begin(vector), std::end(vector)));

  if (!node->isUniform()) {
    auto const& elements = results(std::begin(vector), std::end(vector));
    auto vectorVal = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    this->vectorsByNode[node][vectorIndex] = vectorVal.getDefiningOp();
    return success();
  }

  SmallVector<Attribute, 4> constants;
  for (auto* vectorOp : vector) {
    constants.emplace_back(static_cast<ConstantOp>(vectorOp).getValue());
  }
  auto const& elements = denseFloatingPoints(std::begin(constants), std::end(constants), vectorType);

  auto constVector = rewriter.create<mlir::ConstantOp>(constantOp->getLoc(), elements);

  this->vectorsByNode[node][vectorIndex] = constVector;

  return success();
}

LogicalResult VectorizeBatchRead::matchAndRewrite(SPNBatchRead batchReadOp, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(batchReadOp);

  auto const& vectorIndex = node->getVectorIndex(batchReadOp);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), batchReadOp.getType());

  rewriter.setInsertionPoint(firstUser(std::begin(vector), std::end(vector)));

  if (!node->isUniform() || !areConsecutiveLoads(vector)) {
    auto const& elements = results(std::begin(vector), std::end(vector));
    auto vectorVal = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    vectorsByNode[node][vectorIndex] = vectorVal.getDefiningOp();
  } else {

    auto batchReadLoc = batchReadOp->getLoc();
    auto memIndex = rewriter.create<ConstantOp>(batchReadLoc, rewriter.getIndexAttr(batchReadOp.sampleIndex()));
    ValueRange indices{batchReadOp.batchIndex(), memIndex};
    auto vectorLoad = rewriter.create<vector::LoadOp>(batchReadLoc, vectorType, batchReadOp.batchMem(), indices);

    vectorsByNode[node][vectorIndex] = vectorLoad;
  }

  return success();
}

LogicalResult VectorizeAdd::matchAndRewrite(SPNAdd addOp, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(addOp);

  auto const& vectorIndex = node->getVectorIndex(addOp);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), addOp.getType());

  rewriter.setInsertionPoint(firstOperation(std::begin(vector), std::end(vector)));

  if (!node->isUniform()) {
    auto const& elements = results(std::begin(vector), std::end(vector));
    auto vectorVal = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    vectorsByNode[node][vectorIndex] = vectorVal.getDefiningOp();
    return success();
  }

  llvm::SmallVector<Value, 2> operands;

  for (unsigned i = 0; i < addOp.getNumOperands(); ++i) {
    Value operand;
    if (std::any_of(std::begin(vector), std::end(vector), [&](auto* vectorOp) {
      Value vectorOperand = addOp.getOperand(i);
      return vectorOperand.isa<BlockArgument>() || !parentNodes.count(vectorOperand.getDefiningOp());
    })) {
      llvm::SmallVector<Value, 4> elements;
      for (auto* vectorOp : vector) {
        elements.emplace_back(vectorOp->getOperand(i));
      }
      operand = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    } else {
      auto* operandOp = addOp.getOperand(i).getDefiningOp();
      auto* operandNode = parentNodes.lookup(operandOp);
      if (!vectorsByNode.count(operandNode) || vectorsByNode[operandNode].size() != operandNode->numVectors()) {
        return rewriter.notifyMatchFailure(addOp, "operation's LHS has not yet been (fully) vectorized");
      }
      operand = vectorsByNode[operandNode][operandNode->getVectorIndex(operandOp)]->getResult(0);
    }
    operands.emplace_back(operand);
  }

  auto vectorAddOp = rewriter.create<AddFOp>(addOp->getLoc(), vectorType, operands);

  vectorsByNode[node][vectorIndex] = vectorAddOp;

  return success();
}

LogicalResult VectorizeMul::matchAndRewrite(SPNMul mulOp, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(mulOp);

  auto const& vectorIndex = node->getVectorIndex(mulOp);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), mulOp.getType());

  rewriter.setInsertionPoint(firstOperation(std::begin(vector), std::end(vector)));

  if (!node->isUniform()) {
    auto const& elements = results(std::begin(vector), std::end(vector));
    auto vectorVal = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    vectorsByNode[node][vectorIndex] = vectorVal.getDefiningOp();
    return success();
  }

  llvm::SmallVector<Value, 2> operands;

  for (unsigned i = 0; i < mulOp.getNumOperands(); ++i) {
    Value operand;
    if (std::any_of(std::begin(vector), std::end(vector), [&](auto* vectorOp) {
      Value vectorOperand = mulOp.getOperand(i);
      return vectorOperand.isa<BlockArgument>() || !parentNodes.count(vectorOperand.getDefiningOp());
    })) {
      llvm::SmallVector<Value, 4> elements;
      for (auto* vectorOp : vector) {
        elements.emplace_back(vectorOp->getOperand(i));
      }
      operand = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    } else {
      auto* operandOp = mulOp.getOperand(i).getDefiningOp();
      auto* operandNode = parentNodes.lookup(operandOp);
      if (!vectorsByNode.count(operandNode) || vectorsByNode[operandNode].size() != operandNode->numVectors()) {
        return rewriter.notifyMatchFailure(mulOp, "operation's LHS has not yet been (fully) vectorized");
      }
      operand = vectorsByNode[operandNode][operandNode->getVectorIndex(operandOp)]->getResult(0);
    }
    operands.emplace_back(operand);
  }

  auto vectorMulOp = rewriter.create<MulFOp>(mulOp->getLoc(), vectorType, operands);

  vectorsByNode[node][vectorIndex] = vectorMulOp;

  return success();
}

LogicalResult VectorizeGaussian::matchAndRewrite(SPNGaussianLeaf gaussianOp, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(gaussianOp);

  auto const& vectorIndex = node->getVectorIndex(gaussianOp);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), gaussianOp.getType());

  rewriter.setInsertionPoint(firstOperation(std::begin(vector), std::end(vector)));

  if (!node->isUniform()) {
    auto const& elements = results(std::begin(vector), std::end(vector));
    auto vectorVal = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    vectorsByNode[node][vectorIndex] = vectorVal.getDefiningOp();
    return success();
  }

  DenseElementsAttr coefficients;
  if (vectorType.getElementType().cast<FloatType>().getWidth() == 32) {
    SmallVector<float, 4> array;
    for (auto* vectorOp : vector) {
      float stddev = static_cast<SPNGaussianLeaf>(vectorOp).stddev().convertToFloat();
      array.emplace_back(1.0f / (stddev * std::sqrt(2.0f * M_PIf32)));
    }
    coefficients = DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<float>>(array));
  } else {
    SmallVector<double, 4> array;
    for (auto* vectorOp : vector) {
      double stddev = static_cast<SPNGaussianLeaf>(vectorOp).stddev().convertToDouble();
      array.emplace_back(1.0 / (stddev * std::sqrt(2.0 * M_PI)));
    }
    coefficients = DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<double>>(array));
  }

  // Gather means in a dense floating point attribute vector.
  SmallVector<Attribute, 4> meanAttributes;
  for (auto* vectorOp : vector) {
    meanAttributes.emplace_back(static_cast<SPNGaussianLeaf>(vectorOp).meanAttr());
  }
  auto const& means = denseFloatingPoints(std::begin(meanAttributes), std::end(meanAttributes), vectorType);

  // Gather standard deviations in a dense floating point attribute vector.
  SmallVector<Attribute, 4> stddevAttributes;
  for (auto* vectorOp : vector) {
    stddevAttributes.emplace_back(static_cast<SPNGaussianLeaf>(vectorOp).stddevAttr());
  }
  auto const& stddevs = denseFloatingPoints(std::begin(stddevAttributes), std::end(stddevAttributes), vectorType);

  // Grab the input vector.
  Value inputVector;
  if (std::any_of(std::begin(vector), std::end(vector), [&](auto* vectorOp) {
    return static_cast<SPNGaussianLeaf>(vectorOp).index().template isa<BlockArgument>();
  })) {
    llvm::SmallVector<Value, 4> elements;
    for (auto* vectorOp : vector) {
      elements.emplace_back(static_cast<SPNGaussianLeaf>(vectorOp).index());
    }
    inputVector = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
  } else {
    auto* inputOp = gaussianOp.index().getDefiningOp();
    auto* inputNode = parentNodes.lookup(inputOp);
    if (!vectorsByNode.count(inputNode) || vectorsByNode[inputNode].size() != inputNode->numVectors()) {
      return rewriter.notifyMatchFailure(gaussianOp, "operation's input node has not yet been (fully) vectorized");
    }
    inputVector = vectorsByNode[inputNode][inputNode->getVectorIndex(inputOp)]->getResult(0);
  }

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

  vectorsByNode[node][vectorIndex] = gaussianVector.getDefiningOp();

  return success();
}
