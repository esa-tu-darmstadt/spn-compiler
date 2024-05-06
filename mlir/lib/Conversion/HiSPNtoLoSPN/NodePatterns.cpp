//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "HiSPNtoLoSPN/NodePatterns.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::high;

Value ProductNodeLowering::splitProduct(
    high::ProductNode op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (operands.size() == 1) {
    return operands[0];
  }
  if (operands.size() == 2) {
    return rewriter.create<low::SPNMul>(op.getLoc(), operands[0], operands[1]);
  }
  auto pivot = llvm::divideCeil(operands.size(), 2);
  SmallVector<Value, 10> leftOperands;
  SmallVector<Value, 10> rightOperands;
  unsigned count = 0;
  for (auto &v : operands) {
    if (count < pivot) {
      leftOperands.push_back(v);
    } else {
      rightOperands.push_back(v);
    }
    ++count;
  }
  auto leftTree = splitProduct(op, leftOperands, rewriter);
  auto rightTree = splitProduct(op, rightOperands, rewriter);
  return rewriter.create<low::SPNMul>(op.getLoc(), leftTree, rightTree);
}

LogicalResult ProductNodeLowering::matchAndRewriteChecked(
    high::ProductNode op, high::ProductNode::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  mlir::ValueRange operands = adaptor.getOperands();
  SmallVector<Value, 10> operandsVec(operands.begin(), operands.end());
  rewriter.replaceOp(op, {splitProduct(op, operandsVec, rewriter)});
  return success();
}

Value SumNodeLowering::splitWeightedSum(
    high::SumNode op, ArrayRef<Value> operands, ArrayRef<double> weights,
    ConversionPatternRewriter &rewriter) const {
  if (operands.size() == 1) {
    assert(weights.size() == 1 &&
           "Expecting identical number of operands and weights");
    auto type = typeConverter->convertType(op.getType());
    double weight = weights[0];
    if (type.isa<low::LogType>()) {
      weight = log(weight);
    }
    auto constant = rewriter.create<low::SPNConstant>(
        op.getLoc(), type, rewriter.getF64FloatAttr(weight));

    return rewriter.create<low::SPNMul>(op.getLoc(), operands[0], constant);
  } else {
    auto pivot = llvm::divideCeil(operands.size(), 2);
    SmallVector<Value, 10> leftOperands;
    SmallVector<Value, 10> rightOperands;
    SmallVector<double, 10> leftWeights;
    SmallVector<double, 10> rightWeights;
    unsigned count = 0;
    for (auto ov : llvm::zip(operands, weights)) {
      auto addend = std::get<0>(ov);
      auto weight = std::get<1>(ov);
      if (count < pivot) {
        leftOperands.push_back(addend);
        leftWeights.push_back(weight);
      } else {
        rightOperands.push_back(addend);
        rightWeights.push_back(weight);
      }
      ++count;
    }
    auto leftTree = splitWeightedSum(op, leftOperands, leftWeights, rewriter);
    auto rightTree =
        splitWeightedSum(op, rightOperands, rightWeights, rewriter);
    return rewriter.create<low::SPNAdd>(op->getLoc(), leftTree, rightTree);
  }
}

LogicalResult SumNodeLowering::matchAndRewriteChecked(
    high::SumNode op, high::SumNode::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  SmallVector<double, 10> weights;
  for (auto w : op.getWeights().getValue()) {
    weights.push_back(w.cast<FloatAttr>().getValueAsDouble());
  }
  auto operands = adaptor.getOperands();
  SmallVector<Value, 10> operandsVec(operands.begin(), operands.end());
  rewriter.replaceOp(op,
                     {splitWeightedSum(op, operandsVec, weights, rewriter)});
  return success();
}

LogicalResult HistogramNodeLowering::matchAndRewriteChecked(
    high::HistogramNode op, high::HistogramNode::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // We can safely cast here, as the pattern checks for the correct type of the
  // enclosing query beforehand.
  auto supportMarginal =
      cast<JointQuery>(op.getEnclosingQuery()).getSupportMarginal();
  rewriter.replaceOpWithNewOp<low::SPNHistogramLeaf>(
      op, typeConverter->convertType(op.getType()), op.getIndex(),
      op.getBuckets(), op.getBucketCount(), supportMarginal);
  return success();
}

LogicalResult CategoricalNodeLowering::matchAndRewriteChecked(
    high::CategoricalNode op, high::CategoricalNode::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // We can safely cast here, as the pattern checks for the correct type of the
  // enclosing query beforehand.
  auto supportMarginal =
      cast<JointQuery>(op.getEnclosingQuery()).getSupportMarginal();
  rewriter.replaceOpWithNewOp<low::SPNCategoricalLeaf>(
      op, typeConverter->convertType(op.getType()), op.getIndex(),
      op.getProbabilities(), supportMarginal);
  return success();
}

LogicalResult GaussianNodeLowering::matchAndRewriteChecked(
    high::GaussianNode op, high::GaussianNode::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // We can safely cast here, as the pattern checks for the correct type of the
  // enclosing query beforehand.
  auto supportMarginal =
      cast<JointQuery>(op.getEnclosingQuery()).getSupportMarginal();
  rewriter.replaceOpWithNewOp<low::SPNGaussianLeaf>(
      op, typeConverter->convertType(op.getType()), op.getIndex(), op.getMean(),
      op.getStddev(), supportMarginal);
  return success();
}

namespace {

bool isLogType(Type type) { return type.isa<low::LogType>(); }

} // namespace

LogicalResult RootNodeLowering::matchAndRewriteChecked(
    high::RootNode op, high::RootNode::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto operands = adaptor.getOperands();
  assert(operands.size() == 1 &&
         "Expecting only a single result for a JointQuery");
  Value result = operands[0];
  if (!isLogType(result.getType())) {
    // Insert a conversion to log before returning the result.
    // Currently always uses F64 type to represent log results.
    result = rewriter.create<low::SPNLog>(op->getLoc(), operands[0].getType(),
                                          result);
  }
  rewriter.replaceOpWithNewOp<low::SPNYield>(op, result);
  return success();
}
