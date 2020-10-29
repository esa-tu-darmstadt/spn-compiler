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
  static int histCount = 0;

  assert(operands.size() == 1);

  // Collect all mappings from input var value to probability value in a map.
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

  // Flatten the map into an array by filling up empty indices with 0 values.
  SmallVector<Attribute, 256> valArray;
  for (int i = 0; i < maxUB; ++i) {
    if (values.count(i)) {
      // TODO Make type flexible based on analysis
      valArray.push_back(rewriter.getF64FloatAttr(values[i]));
    } else {
      valArray.push_back(rewriter.getF64FloatAttr(0));
    }
  }
  auto valArrayAttr = rewriter.getArrayAttr(valArray);

  auto doubleType = LLVM::LLVMType::getDoubleTy(op.getContext());
  auto arrType = LLVM::LLVMType::getArrayTy(doubleType, maxUB);
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

  auto addressOf = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), globalConst);
  auto indexType = LLVM::LLVMType::getInt64Ty(rewriter.getContext());
  auto constZeroIndex = rewriter.create<LLVM::ConstantOp>(op.getLoc(), indexType, rewriter.getI64IntegerAttr(0));
  auto ptrType = doubleType.getPointerTo();
  auto gep = rewriter.create<LLVM::GEPOp>(op.getLoc(), ptrType, addressOf, ValueRange{constZeroIndex, operands[0]});
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, gep);
  return success();
}
