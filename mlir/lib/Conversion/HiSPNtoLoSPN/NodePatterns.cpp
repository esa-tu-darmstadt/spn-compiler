//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "HiSPNtoLoSPN/NodePatterns.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNTypes.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::high;

Value ProductNodeLowering::splitProduct(high::ProductNode op, ArrayRef<Value> operands,
                                        ConversionPatternRewriter& rewriter) const {
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
  for (auto& v : operands) {
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

LogicalResult ProductNodeLowering::matchAndRewriteChecked(high::ProductNode op,
                                                          ArrayRef<Value> operands,
                                                          ConversionPatternRewriter& rewriter) const {
  rewriter.replaceOp(op, {splitProduct(op, operands, rewriter)});
  return success();
}

Value SumNodeLowering::splitWeightedSum(high::SumNode op,
                                        ArrayRef<Value> operands,
                                        ArrayRef<double> weights,
                                        ConversionPatternRewriter& rewriter) const {
  if (operands.size() == 1) {
    assert(weights.size() == 1 && "Expecting identical number of operands and weights");
    auto type = typeConverter->convertType(op.getType());
    double weight = weights[0];
    if (type.isa<low::LogType>()) {
      weight = log(weight);
    }
    auto constant = rewriter.create<low::SPNConstant>(op.getLoc(), type,
                                                      TypeAttr::get(type),
                                                      rewriter.getF64FloatAttr(weight));

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
    auto rightTree = splitWeightedSum(op, rightOperands, rightWeights, rewriter);
    return rewriter.create<low::SPNAdd>(op->getLoc(), leftTree, rightTree);
  }
}

LogicalResult SumNodeLowering::matchAndRewriteChecked(high::SumNode op,
                                                      ArrayRef<Value> operands,
                                                      ConversionPatternRewriter& rewriter) const {
  SmallVector<double, 10> weights;
  for (auto w : op.weights().getValue()) {
    weights.push_back(w.cast<FloatAttr>().getValueAsDouble());
  }
  rewriter.replaceOp(op, {splitWeightedSum(op, operands, weights, rewriter)});
  return success();
}

LogicalResult HistogramNodeLowering::matchAndRewriteChecked(high::HistogramNode op,
                                                            ArrayRef<Value> operands,
                                                            ConversionPatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<low::SPNHistogramLeaf>(op, typeConverter->convertType(op.getType()),
                                                     op.index(), op.buckets(),
                                                     op.bucketCount(), false);
  return success();
}

LogicalResult CategoricalNodeLowering::matchAndRewriteChecked(high::CategoricalNode op,
                                                              ArrayRef<Value> operands,
                                                              ConversionPatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<low::SPNCategoricalLeaf>(op, typeConverter->convertType(op.getType()),
                                                       op.index(), op.probabilities(), false);
  return success();
}

LogicalResult GaussianNodeLowering::matchAndRewriteChecked(high::GaussianNode op,
                                                           ArrayRef<Value> operands,
                                                           ConversionPatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<low::SPNGaussianLeaf>(op, typeConverter->convertType(op.getType()),
                                                    op.index(), op.mean(), op.stddev(), false);
  return success();
}

namespace {

  bool isLogType(Type type) {
    return type.isa<low::LogType>();
  }

}

LogicalResult RootNodeLowering::matchAndRewriteChecked(high::RootNode op,
                                                       ArrayRef<Value> operands,
                                                       ConversionPatternRewriter& rewriter) const {
  assert(operands.size() == 1 && "Expecting only a single result for a JointQuery");
  Value result = operands[0];
  if (!isLogType(result.getType())) {
    // Insert a conversion to log before returning the result.
    // Currently always uses F64 type to represent log results.
    result = rewriter.create<low::SPNLog>(op->getLoc(), rewriter.getF64Type(), result);
  }
  rewriter.replaceOpWithNewOp<low::SPNYield>(op, result);
  return success();
}
