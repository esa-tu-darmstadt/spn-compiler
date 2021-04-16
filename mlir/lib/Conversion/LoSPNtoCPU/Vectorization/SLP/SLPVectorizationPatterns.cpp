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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "LoSPN/LoSPNAttributes.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

// Helper functions in anonymous namespace.
namespace {

  void broadcastIdentical(Value const& source, Type const& vectorType, PatternRewriter& rewriter) {
    auto const& insertionPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(source);
    rewriter.create<vector::BroadcastOp>(source.getLoc(), vectorType, source);
    rewriter.restoreInsertionPoint(insertionPoint);
  }

  template<typename V>
  Value broadcastFirstInsertRest(V const& vector, VectorType const& vectorType, PatternRewriter& rewriter) {
    llvm::SmallVector<Value, 4> elements;
    for (size_t i = 0; i < vector.size(); ++i) {
      elements.template emplace_back(vector[i]->getResult(0));
    }
    auto const& insertionPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(elements[0]);
    Value vectorOp = rewriter.create<vector::BroadcastOp>(vector[0]->getLoc(), vectorType, elements[0]);
    for (size_t i = 1; i < vector.size(); ++i) {
      auto const& position = i/*getOrCreateConstant(i, false)*/;
      vectorOp = rewriter.create<vector::InsertElementOp>(vector[0]->getLoc(), elements[i], vectorOp, position);
    }
    rewriter.restoreInsertionPoint(insertionPoint);
    return vectorOp;
  }
}

LogicalResult VectorizeBatchRead::matchAndRewrite(SPNBatchRead op, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(op);
  node->dump();

  auto const& vectorIndex = node->getVectorIndex(op);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), op.getType());

  if (!node->isUniform()) {
    //vectorsByNode[node][vectorIndex] = broadcastFirstInsertRest(vector, vectorType, rewriter).getDefiningOp();
    return success();
  }
  return success();
}

LogicalResult VectorizeAdd::matchAndRewrite(SPNAdd op, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(op);
  node->dump();

  auto const& vectorIndex = node->getVectorIndex(op);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), op.getType());

  if (!node->isUniform()) {
    //vectorsByNode[node][vectorIndex] = broadcastFirstInsertRest(vector, vectorType, rewriter).getDefiningOp();
    return success();
  }
  return success();
}

LogicalResult VectorizeMul::matchAndRewrite(SPNMul op, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(op);
  node->dump();

  auto const& vectorIndex = node->getVectorIndex(op);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), op.getType());

  if (!node->isUniform()) {
    //vectorsByNode[node][vectorIndex] = broadcastFirstInsertRest(vector, vectorType, rewriter).getDefiningOp();
    return success();
  }
  return success();
}

LogicalResult VectorizeLog::matchAndRewrite(SPNLog op, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(op);
  node->dump();

  auto const& vectorIndex = node->getVectorIndex(op);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), op.getType());

  if (!node->isUniform()) {
    //vectorsByNode[node][vectorIndex] = broadcastFirstInsertRest(vector, vectorType, rewriter).getDefiningOp();
    return success();
  }

  return success();
}

LogicalResult VectorizeConstant::matchAndRewrite(SPNConstant op, PatternRewriter& rewriter) const {

  auto* node = parentNodes.lookup(op);
  node->dump();

  auto const& vectorIndex = node->getVectorIndex(op);
  auto const& vector = node->getVector(vectorIndex);
  auto const& vectorType = VectorType::get(static_cast<unsigned>(vector.size()), op.getType());

  if (!node->isUniform()) {
    vectorsByNode[node][vectorIndex] = broadcastFirstInsertRest(vector, vectorType, rewriter).getDefiningOp();
    return success();
  }

  DenseElementsAttr constAttr;

  if (vectorType.getElementType().cast<FloatType>().getWidth() == 32) {
    llvm::SmallVector<float, 4> array;
    for (int i = 0; i < vectorType.getNumElements(); ++i) {
      array.push_back(static_cast<SPNConstant>(vector[i]).value().convertToFloat());
    }
    constAttr = mlir::DenseElementsAttr::get(vectorType, (llvm::ArrayRef<float>) array);
  } else {
    llvm::SmallVector<double, 4> array;
    for (int i = 0; i < vectorType.getNumElements(); ++i) {
      array.push_back(static_cast<SPNConstant>(vector[i]).value().convertToDouble());
    }
    constAttr = mlir::DenseElementsAttr::get(vectorType, (llvm::ArrayRef<double>) array);
  }

  auto const& insertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointAfter(vector.front());
  auto constValue = rewriter.create<mlir::ConstantOp>(op->getLoc(), constAttr);
  rewriter.restoreInsertionPoint(insertionPoint);

  vectorsByNode[node][vectorIndex] = constValue;

  constValue->dump();

  return success();
}
