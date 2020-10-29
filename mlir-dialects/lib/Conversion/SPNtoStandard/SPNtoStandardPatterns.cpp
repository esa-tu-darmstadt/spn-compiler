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
  return failure();
}

mlir::LogicalResult mlir::spn::SingleJointLowering::matchAndRewrite(mlir::spn::SingleJointQuery op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  auto inputType = MemRefType::get({op.numFeatures()}, op.inputType());
  auto returnOp = op.graph().front().getTerminator();
  auto graphResult = dyn_cast<mlir::spn::ReturnOp>(returnOp);
  assert(graphResult);
  graphResult.dump();
  auto resultType = MemRefType::get({1}, graphResult.retValue().front().getType());

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
  rewriter.setInsertionPointToEnd(funcEntryBlock);
  SmallVector<Value, 1> indices;
  indices.push_back(rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0)));
  rewriter.create<mlir::StoreOp>(op.getLoc(), graphResult.retValue().front(),
                                 replaceFunc.getArgument(1), indices);
  rewriter.create<mlir::ReturnOp>(op.getLoc());
  rewriter.eraseOp(graphResult);
  rewriter.eraseOp(op);
  return success();
}
