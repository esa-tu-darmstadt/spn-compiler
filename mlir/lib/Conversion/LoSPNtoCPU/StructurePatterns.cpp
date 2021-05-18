//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <mlir/IR/BlockAndValueMapping.h>
#include "LoSPNtoCPU/StructurePatterns.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

mlir::LogicalResult mlir::spn::KernelLowering::matchAndRewrite(mlir::spn::low::SPNKernel op,
                                                               llvm::ArrayRef<mlir::Value> operands,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.empty() && "Kernel should not take any operands");
  auto replaceFunc = rewriter.create<mlir::FuncOp>(op.getLoc(), op.getName(), op.getType());
  auto funcBlock = replaceFunc.addEntryBlock();
  rewriter.mergeBlocks(&op.body().front(), funcBlock, funcBlock->getArguments());
  rewriter.eraseOp(op);
  return success();
}

mlir::LogicalResult mlir::spn::BatchTaskLowering::matchAndRewrite(mlir::spn::low::SPNTask op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  //
  // Lower a task with batchSize > 1. The task is lowered to a function, containing a scalar loop iterating over the
  // samples in the batch. The content of the Task is merged into the newly created loop's body, the loop induction
  // variable replaces the batchIndex argument of the Task.
  static int taskCount = 0;
  if (op.batchSize() == 1) {
    return rewriter.notifyMatchFailure(op, "Match only batched (batchSize > 1) execution");
  }
  auto restore = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(op->getParentOfType<mlir::ModuleOp>().getBody());
  SmallVector<Type, 5> inputTypes;
  for (auto operand : operands) {
    inputTypes.push_back(operand.getType());
  }
  auto funcType = FunctionType::get(rewriter.getContext(), inputTypes, {});
  auto taskFunc = rewriter.create<FuncOp>(op->getLoc(), Twine("task_", std::to_string(taskCount++)).str(),
                                          funcType);
  auto taskBlock = taskFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(taskBlock);
  auto const0 = rewriter.create<ConstantOp>(op->getLoc(), rewriter.getIndexAttr(0));
  auto ub = rewriter.create<memref::DimOp>(op.getLoc(), taskBlock->getArgument(0), 0);
  auto step = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1));
  auto loop = rewriter.create<scf::ForOp>(op.getLoc(), const0, ub, step);
  rewriter.create<ReturnOp>(op->getLoc());
  // Fill the loop
  auto& loopBlock = loop.getLoopBody().front();
  rewriter.setInsertionPointToStart(&loopBlock);
  // Collect the values replacing the block values of old block inside the task.
  // The first argument is the batch index, i.e., the loop induction var.
  // The other arguments are the arguments of the entry block of this function.
  SmallVector<Value, 5> blockReplacementArgs;
  blockReplacementArgs.push_back(loop.getInductionVar());
  for (auto bArg : taskBlock->getArguments()) {
    blockReplacementArgs.push_back(bArg);
  }
  rewriter.mergeBlockBefore(&op.body().front(), loopBlock.getTerminator(), blockReplacementArgs);
  loopBlock.walk([&rewriter](low::SPNReturn ret) {
    assert(ret.returnValues().empty() && "Task return should be empty");
    rewriter.eraseOp(ret);
  });
  // Insert a call to the newly created task function.
  rewriter.restoreInsertionPoint(restore);
  rewriter.replaceOpWithNewOp<mlir::CallOp>(op, taskFunc, operands);
  return success();
}

mlir::LogicalResult mlir::spn::SingleTaskLowering::matchAndRewrite(mlir::spn::low::SPNTask op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  //
  // Lower a task with batchSize == 1. The task is lowered to a function, the content of the Task is merged into the
  // newly created function. As only a single execution is required, the batchIndex argument of the body can
  // be replaced with a constant zero.
  static int taskCount = 0;
  if (op.batchSize() != 1) {
    return rewriter.notifyMatchFailure(op, "Match only single (batchSize == 1) execution");
  }

  auto restore = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(op->getParentOfType<mlir::ModuleOp>().getBody());
  SmallVector<Type, 5> inputTypes;
  for (auto operand : operands) {
    inputTypes.push_back(operand.getType());
  }
  auto funcType = FunctionType::get(rewriter.getContext(), inputTypes, {});
  auto taskFunc = rewriter.create<FuncOp>(op->getLoc(), Twine("task_", std::to_string(taskCount++)).str(), funcType);
  auto taskBlock = taskFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(taskBlock);

  // Collect the values replacing the block values of old block inside the task.
  // The first argument is the batch index, in this case (for a single execution),
  // we can simply set it to constant zero.
  // The other arguments are the arguments of the entry block of this function.
  SmallVector<Value, 5> blockReplacementArgs;
  blockReplacementArgs.push_back(rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0)));
  for (auto bArg : taskBlock->getArguments()) {
    blockReplacementArgs.push_back(bArg);
  }
  // Inline the content of the Task into the function.
  rewriter.mergeBlocks(&op.body().front(), taskBlock, blockReplacementArgs);
  // Insert a call to the newly created task function.
  rewriter.restoreInsertionPoint(restore);
  rewriter.replaceOpWithNewOp<mlir::CallOp>(op, taskFunc, operands);
  return success();
}

mlir::LogicalResult mlir::spn::BodyLowering::matchAndRewrite(mlir::spn::low::SPNBody op,
                                                             llvm::ArrayRef<mlir::Value> operands,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.size() == op.body().front().getNumArguments() &&
      "Expecting the number of operands to match the block arguments");

  SmallVector<Value, 2> resultValues;
  op.body().front().walk([&](low::SPNYield yield) {
    for (auto res : yield.resultValues()) {
      if (auto logType = res.getType().dyn_cast<low::LogType>()) {
        // If the body internally computes in log-space, we need to
        // strip the log-semantic for the operations using the result of the body.
        rewriter.setInsertionPoint(yield);
        auto stripLog = rewriter.create<low::SPNStripLog>(yield->getLoc(), res, logType.getBaseType());
        if (auto vectorizedRes = dyn_cast<low::LoSPNVectorizable>(res.getDefiningOp())) {
          if (vectorizedRes.checkVectorized()) {
            stripLog.setVectorized(vectorizedRes.getVectorWidth());
          }
        }
        resultValues.push_back(stripLog);
      } else {
        resultValues.push_back(res);
      }
    }
    rewriter.eraseOp(yield);
  });
  rewriter.mergeBlockBefore(&op.body().front(), op, operands);
  rewriter.replaceOp(op, resultValues);
  return success();
}
