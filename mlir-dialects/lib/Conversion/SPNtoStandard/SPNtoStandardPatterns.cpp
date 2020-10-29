//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <mlir/IR/Function.h>
#include "mlir/IR/Module.h"
#include "SPNtoStandard/SPNtoStandardPatterns.h"
#include "mlir/IR/BlockAndValueMapping.h"

mlir::LogicalResult mlir::spn::ConstantOpLowering::matchAndRewrite(mlir::spn::ConstantOp op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, op.valueAttr());
  return success();
}

mlir::LogicalResult mlir::spn::ReturnOpLowering::matchAndRewrite(mlir::spn::ReturnOp op,
                                                                 llvm::ArrayRef<mlir::Value> operands,
                                                                 mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.size() == 1);
  rewriter.replaceOp(op, operands[0]);
  return success();
}

mlir::LogicalResult mlir::spn::SingleJointLowering::matchAndRewrite(mlir::spn::SingleJointQuery op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  auto inputType = MemRefType::get({op.numFeatures()}, op.inputType());
  // TODO Currently simply assumes F64 type, add logic for different types as necessary.
  auto resultType = MemRefType::get({1}, rewriter.getF64Type());

  auto replaceFunc = rewriter.create<FuncOp>(op.getLoc(), "single_joint",
                                             rewriter.getFunctionType({inputType, resultType}, llvm::None),
                                             llvm::None);

  auto funcEntryBlock = replaceFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(funcEntryBlock);
  auto inputArg = replaceFunc.getArgument(0);
  assert(inputArg.getType().isa<MemRefType>());
  SmallVector<Value, 10> blockArgsReplacement;
  for (size_t i = 0; i < op.numFeatures(); ++i) {
    SmallVector<Value, 1> indices;
    indices.push_back(rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i)));
    auto load = rewriter.create<mlir::LoadOp>(op.getLoc(), inputArg, indices);
    blockArgsReplacement.push_back(load);
  }
  rewriter.mergeBlocks(&op.getRegion().front(), funcEntryBlock, blockArgsReplacement);

  rewriter.eraseOp(op);
  return success();
}
