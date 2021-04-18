//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/IR/BlockAndValueMapping.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
#include "LoSPNtoCPU/Vectorization/TargetInformation.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "LoSPN/LoSPNAttributes.h"

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
    auto const& insertionPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(*begin);
    Value vectorOp = rewriter.create<vector::BroadcastOp>(begin->getLoc(), vectorType, *begin);
    unsigned position = 1;
    while (++begin != end) {
      rewriter.setInsertionPointAfterValue(*begin);
      vectorOp = rewriter.create<vector::InsertElementOp>(begin->getLoc(), *begin, vectorOp, position++);
    }
    rewriter.restoreInsertionPoint(insertionPoint);
    return vectorOp;
  }

  template<typename OperationIterator>
  llvm::SmallVector<Value, 4> results(OperationIterator begin, OperationIterator end) {
    llvm::SmallVector<Value, 4> elements;
    while (begin != end) {
      elements.emplace_back((*begin)->getResult(0));
      ++begin;
    }
    return elements;
  }

  template<typename OperationIterator>
  Operation* firstOperation(OperationIterator begin, OperationIterator end) {
    Operation* firstOp = *begin;
    while (++begin != end) {
      if (!firstOp->isBeforeInBlock(*begin)) {
        firstOp = *begin;
      }
    }
    return firstOp;
  }

  template<typename ValueIterator>
  Value firstValue(ValueIterator begin, ValueIterator end) {
    Value firstVal = *begin;
    while (begin != end) {
      if (begin->template isa<BlockArgument>()) {
        return *begin;
      }
      if (!firstVal.getDefiningOp()->isBeforeInBlock(*begin)) {
        firstVal = *begin;
      }
      ++begin;
    }
    return firstVal;
  }

  template<typename OperationIterator>
  Operation* lastOperation(OperationIterator begin, OperationIterator end) {
    Operation* lastOp = *begin;
    while (++begin != end) {
      if (lastOp->isBeforeInBlock(*begin)) {
        lastOp = *begin;
      }
    }
    return lastOp;
  }

  template<typename ValueIterator>
  Value lastValue(ValueIterator begin, ValueIterator end) {
    Value lastVal = *begin;
    while (++begin != end) {
      if (begin->template isa<BlockArgument>()) {
        continue;
      } else if (lastVal.isa<BlockArgument>() || lastVal.getDefiningOp()->isBeforeInBlock(begin->getDefiningOp())) {
        lastVal = *begin;
      }
    }
    return lastVal;
  }

}

template<typename ConstantSourceOp>
LogicalResult VectorizeConstantPattern<ConstantSourceOp>::matchAndRewrite(ConstantSourceOp op, PatternRewriter& rewriter) const {

  if(!op->template hasTrait<OpTrait::ConstantLike>()) {
    rewriter.notifyMatchFailure(op, "Constant vectorization pattern cannot be applied to this operation");
  }

  auto* node = this->parentNodes.lookup(op);

  auto const& vectorIndex = node->getVectorIndex(op);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), op.getType());

  if (!node->isUniform()) {
    auto const& elements = results(std::begin(vector), std::end(vector));
    auto vectorVal = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    this->vectorsByNode[node][vectorIndex] = vectorVal.getDefiningOp();
    return success();
  }

  DenseElementsAttr constAttr;

  if (vectorType.getElementType().template cast<FloatType>().getWidth() == 32) {
    llvm::SmallVector<float, 4> array;
    for (int i = 0; i < vectorType.getNumElements(); ++i) {
      array.push_back(static_cast<SPNConstant>(vector[i]).value().convertToFloat());
    }
    constAttr = mlir::DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<float>>(array));
  } else {
    llvm::SmallVector<double, 4> array;
    for (int i = 0; i < vectorType.getNumElements(); ++i) {
      array.push_back(static_cast<SPNConstant>(vector[i]).value().convertToDouble());
    }
    constAttr = mlir::DenseElementsAttr::get(vectorType, static_cast<llvm::ArrayRef<double>>(array));
  }

  rewriter.setInsertionPointAfter(firstOperation(std::begin(vector), std::end(vector)));
  auto constValue = rewriter.create<mlir::ConstantOp>(op->getLoc(), constAttr);

  this->vectorsByNode[node][vectorIndex] = constValue;

  return success();
}

LogicalResult VectorizeBatchRead::matchAndRewrite(SPNBatchRead op, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(op);

  auto const& vectorIndex = node->getVectorIndex(op);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), op.getType());

  if (!node->isUniform() || !areConsecutiveLoads(vector)) {
    auto const& elements = results(std::begin(vector), std::end(vector));
    auto vectorVal = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    vectorsByNode[node][vectorIndex] = vectorVal.getDefiningOp();
  } else {

    rewriter.setInsertionPointAfter(firstOperation(std::begin(vector), std::end(vector)));

    auto memIndex = rewriter.create<ConstantOp>(op->getLoc(), rewriter.getIndexAttr(op.sampleIndex()));
    ValueRange indices{op.batchIndex(), memIndex};

    auto vectorLoad = rewriter.create<vector::LoadOp>(op->getLoc(), vectorType, op.batchMem(), indices);

    vectorsByNode[node][vectorIndex] = vectorLoad;
  }

  return success();
}

LogicalResult VectorizeAdd::matchAndRewrite(SPNAdd op, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(op);

  auto const& vectorIndex = node->getVectorIndex(op);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), op.getType());

  if (!node->isUniform()) {
    auto const& elements = results(std::begin(vector), std::end(vector));
    auto vectorVal = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    vectorsByNode[node][vectorIndex] = vectorVal.getDefiningOp();
    return success();
  }

  llvm::SmallVector<Value, 2> operands;

  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    Value operand;
    if (std::any_of(std::begin(vector), std::end(vector), [&](auto* vectorOp) {
      Value vectorOperand = op.getOperand(i);
      return vectorOperand.isa<BlockArgument>() || !parentNodes.count(vectorOperand.getDefiningOp());
    })) {
      llvm::SmallVector<Value, 4> elements;
      for (auto* vectorOp : vector) {
        elements.emplace_back(vectorOp->getOperand(i));
      }
      operand = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    } else {
      auto* operandOp = op.getOperand(i).getDefiningOp();
      auto* operandNode = parentNodes.lookup(operandOp);
      if (!vectorsByNode.count(operandNode) || vectorsByNode[operandNode].size() != operandNode->numVectors()) {
        return rewriter.notifyMatchFailure(op, "operation's LHS has not yet been (fully) vectorized");
      }
      operand = vectorsByNode[operandNode][operandNode->getVectorIndex(operandOp)]->getResult(0);
    }
    operands.emplace_back(operand);
  }

  rewriter.setInsertionPointAfterValue(lastValue(std::begin(operands), std::end(operands)));
  auto addOp = rewriter.create<AddFOp>(op->getLoc(), vectorType, operands);

  vectorsByNode[node][vectorIndex] = addOp;

  return success();
}

LogicalResult VectorizeMul::matchAndRewrite(SPNMul op, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(op);

  auto const& vectorIndex = node->getVectorIndex(op);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), op.getType());

  if (!node->isUniform()) {
    auto const& elements = results(std::begin(vector), std::end(vector));
    auto vectorVal = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    vectorsByNode[node][vectorIndex] = vectorVal.getDefiningOp();
    return success();
  }

  llvm::SmallVector<Value, 2> operands;

  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    Value operand;
    if (std::any_of(std::begin(vector), std::end(vector), [&](auto* vectorOp) {
      vectorOp->dump();
      Value vectorOperand = op.getOperand(i);
      return vectorOperand.isa<BlockArgument>() || !parentNodes.count(vectorOperand.getDefiningOp());
    })) {
      llvm::SmallVector<Value, 4> elements;
      for (auto* vectorOp : vector) {
        elements.emplace_back(vectorOp->getOperand(i));
      }
      operand = broadcastFirstInsertRest(std::begin(elements), std::end(elements), vectorType, rewriter);
    } else {
      auto* operandOp = op.getOperand(i).getDefiningOp();
      auto* operandNode = parentNodes.lookup(operandOp);
      if (!vectorsByNode.count(operandNode) || vectorsByNode[operandNode].size() != operandNode->numVectors()) {
        return rewriter.notifyMatchFailure(op, "operation's LHS has not yet been (fully) vectorized");
      }
      operand = vectorsByNode[operandNode][operandNode->getVectorIndex(operandOp)]->getResult(0);
    }
    operands.emplace_back(operand);
  }

  rewriter.setInsertionPointAfterValue(lastValue(std::begin(operands), std::end(operands)));
  auto mulOp = rewriter.create<MulFOp>(op->getLoc(), vectorType, operands);

  vectorsByNode[node][vectorIndex] = mulOp;

  return success();
}

LogicalResult VectorizeLog::matchAndRewrite(SPNLog op, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(op);

  auto const& vectorIndex = node->getVectorIndex(op);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), op.getType());

  if (!node->isUniform()) {
    //vectorsByNode[node][vectorIndex] = broadcastFirstInsertRest(vector, vectorType, rewriter).getDefiningOp();
    return success();
  }

  return success();
}
