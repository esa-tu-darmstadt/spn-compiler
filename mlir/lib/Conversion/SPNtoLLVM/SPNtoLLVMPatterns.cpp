//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/IR/BuiltinOps.h"
#include "SPNtoLLVM/SPNtoLLVMPatterns.h"
#include "SPN/SPNAttributes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

mlir::LogicalResult mlir::spn::HistogramOpLowering::matchAndRewrite(mlir::spn::HistogramOp op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  // Check for single operand, i.e. the index value.
  assert(operands.size() == 1);

  // Collect all mappings from input var value to probability value in a map
  // and compute the minimum lower bound & maximum upper bound.
  llvm::DenseMap<int, double> values;
  int minLB = std::numeric_limits<int>::max();
  int maxUB = std::numeric_limits<int>::min();
  for (auto& b : op.bucketsAttr()) {
    auto bucket = b.cast<Bucket>();
    auto lb = bucket.lb().getInt();
    auto ub = bucket.ub().getInt();
    auto val = bucket.val().getValueAsDouble();
    for (int i = lb; i < ub; ++i) {
      values[i] = val;
    }
    minLB = std::min<int>(minLB, lb);
    maxUB = std::max<int>(maxUB, ub);
  }

  // Currently, we assume that all input vars take no values <0.
  if (minLB < 0) {
    return failure();
  }

  auto resultType = op.getResult().getType();
  if (!resultType.isIntOrFloat()) {
    // Currently only handling Int and Float result types.
    return failure();
  }

  // Flatten the map into an array by filling up empty indices with 0 values.
  SmallVector<Attribute, 256> valArray;
  for (int i = 0; i < maxUB; ++i) {
    double indexVal;
    if (values.count(i)) {
      indexVal = values[i];
    } else {
      // Fill up with 0 if no value was defined by the histogram.
      indexVal = 0;
    }
    // Construct attribute with constant value. Need to distinguish cases here due to different builder methods.
    if (resultType.isIntOrIndex()) {
      valArray.push_back(rewriter.getIntegerAttr(resultType, (int) indexVal));
    } else {
      valArray.push_back(rewriter.getFloatAttr(resultType, indexVal));
    }
  }

  return replaceOpWithGlobalArrayLoad<HistogramOp>(op, rewriter, *typeConverter, operands[0], valArray);
}

mlir::LogicalResult mlir::spn::CategoricalOpLowering::matchAndRewrite(mlir::spn::CategoricalOp op,
                                                                      llvm::ArrayRef<mlir::Value> operands,
                                                                      mlir::ConversionPatternRewriter& rewriter) const {
  // Check for single operand, i.e., the index value.
  assert(operands.size() == 1);

  return replaceOpWithGlobalArrayLoad<CategoricalOp>(op, rewriter, *typeConverter,
                                                     operands[0], op.probabilitiesAttr().getValue());
}
