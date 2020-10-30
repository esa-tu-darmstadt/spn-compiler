//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <mlir/IR/Module.h>
#include "SPNtoLLVM/SPNtoLLVMPatterns.h"
#include "SPN/SPNAttributes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

mlir::LogicalResult mlir::spn::HistogramOpLowering::matchAndRewrite(mlir::spn::HistogramOp op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  // Simple count for unique naming of global arrays.
  static int histCount = 0;

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
  auto valArrayAttr = rewriter.getArrayAttr(valArray);

  // Create & insert a constant global array with the values from the histogram.
  auto elementType = typeConverter->convertType(resultType).dyn_cast<mlir::LLVM::LLVMType>();
  assert(elementType);
  auto arrType = LLVM::LLVMType::getArrayTy(elementType, maxUB);
  auto module = op.getParentOfType<ModuleOp>();
  auto restore = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(module.getBody());
  auto globalConst = rewriter.create<LLVM::GlobalOp>(op.getLoc(),
                                                     arrType,
                                                     true,
                                                     LLVM::Linkage::Internal,
                                                     "hist_" + std::to_string(histCount++),
                                                     valArrayAttr);
  rewriter.restoreInsertionPoint(restore);

  // Load a value from the histogram using the index value to index into the histogram.
  auto addressOf = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), globalConst);
  auto indexType = LLVM::LLVMType::getInt64Ty(rewriter.getContext());
  auto constZeroIndex = rewriter.create<LLVM::ConstantOp>(op.getLoc(), indexType, rewriter.getI64IntegerAttr(0));
  auto ptrType = elementType.getPointerTo();
  auto gep = rewriter.create<LLVM::GEPOp>(op.getLoc(), ptrType, addressOf, ValueRange{constZeroIndex, operands[0]});
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, gep);
  return success();
}
