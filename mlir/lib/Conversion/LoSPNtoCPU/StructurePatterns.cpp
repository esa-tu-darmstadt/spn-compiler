//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <mlir/IR/BlockAndValueMapping.h>
#include "LoSPNtoCPU/StructurePatterns.h"
//#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
//#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

mlir::LogicalResult mlir::spn::KernelLowering::matchAndRewrite(mlir::spn::low::SPNKernel op,
                                                               OpAdaptor adaptor,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  assert(operands.empty() && "Kernel should not take any operands");
  auto replaceFunc = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), op.getFunctionType());
  auto funcBlock = replaceFunc.addEntryBlock();
  rewriter.mergeBlocks(&op.getBody().front(), funcBlock, funcBlock->getArguments());
  rewriter.eraseOp(op);
  return success();
}

mlir::LogicalResult mlir::spn::BatchTaskLowering::matchAndRewrite(mlir::spn::low::SPNTask op,
                                                                  OpAdaptor adaptor,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = op.getOperands();
  // Lower a task with batchSize > 1. The task is lowered to a function, containing a scalar loop iterating over the
  // samples in the batch. The content of the Task is merged into the newly created loop's body, the loop induction
  // variable replaces the batchIndex argument of the Task.
  static int taskCount = 0;
  if (op.getBatchSize() == 1) {
    return rewriter.notifyMatchFailure(op, "Match only batched (batchSize > 1) execution");
  }
  auto restore = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(op->getParentOfType<mlir::ModuleOp>().getBody());
  SmallVector<Type, 5> inputTypes;
  for (auto operand : operands) {
    inputTypes.push_back(operand.getType());
  }
  auto funcType = FunctionType::get(rewriter.getContext(), inputTypes, {});
  auto taskFunc = rewriter.create<func::FuncOp>(op->getLoc(), Twine("task_", std::to_string(taskCount++)).str(),
                                          funcType);
  auto taskBlock = taskFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(taskBlock);
  auto const0 = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getIndexAttr(0));
  // The upper bound can be derived from the dynamic dimension of one of the input memrefs.
  auto inputMemRef = taskBlock->getArgument(0);
  auto inputMemRefTy = inputMemRef.getType().dyn_cast<MemRefType>();
  assert(inputMemRefTy);
  assert(inputMemRefTy.hasRank() && inputMemRefTy.getRank() == 2);
  assert(inputMemRefTy.isDynamicDim(0) ^ inputMemRefTy.isDynamicDim(1));
  auto index = (inputMemRefTy.isDynamicDim(0)) ? 0 : 1;
  auto ub = rewriter.create<memref::DimOp>(op.getLoc(), inputMemRef, index);
  auto step = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1));
  auto loop = rewriter.create<scf::ForOp>(op.getLoc(), const0, ub, step);
  rewriter.create<func::ReturnOp>(op->getLoc());
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
  rewriter.mergeBlockBefore(&op.getBody().front(), loopBlock.getTerminator(), blockReplacementArgs);
  loopBlock.walk([&rewriter](low::SPNReturn ret) {
    assert(ret.getReturnValues().empty() && "Task return should be empty");
    rewriter.eraseOp(ret);
  });
  // Insert a call to the newly created task function.
  rewriter.restoreInsertionPoint(restore);
  rewriter.replaceOpWithNewOp<func::CallOp>(op, taskFunc, operands);
  return success();
}

mlir::LogicalResult mlir::spn::SingleTaskLowering::matchAndRewrite(mlir::spn::low::SPNTask op,
                                                                   OpAdaptor adaptor,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  // Lower a task with batchSize == 1. The task is lowered to a function, the content of the Task is merged into the
  // newly created function. As only a single execution is required, the batchIndex argument of the body can
  // be replaced with a constant zero.
  static int taskCount = 0;
  if (op.getBatchSize() != 1) {
    return rewriter.notifyMatchFailure(op, "Match only single (batchSize == 1) execution");
  }

  auto restore = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(op->getParentOfType<mlir::ModuleOp>().getBody());
  SmallVector<Type, 5> inputTypes;
  for (auto operand : operands) {
    inputTypes.push_back(operand.getType());
  }
  auto funcType = FunctionType::get(rewriter.getContext(), inputTypes, {});
  auto taskFunc = rewriter.create<func::FuncOp>(op->getLoc(), Twine("task_", std::to_string(taskCount++)).str(), funcType);
  auto taskBlock = taskFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(taskBlock);

  // Collect the values replacing the block values of old block inside the task.
  // The first argument is the batch index, in this case (for a single execution),
  // we can simply set it to constant zero.
  // The other arguments are the arguments of the entry block of this function.
  SmallVector<Value, 5> blockReplacementArgs;
  blockReplacementArgs.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0)));
  for (auto bArg : taskBlock->getArguments()) {
    blockReplacementArgs.push_back(bArg);
  }
  // Inline the content of the Task into the function.
  rewriter.mergeBlocks(&op.getBody().front(), taskBlock, blockReplacementArgs);
  // Insert a call to the newly created task function.
  rewriter.restoreInsertionPoint(restore);
  rewriter.replaceOpWithNewOp<func::CallOp>(op, taskFunc, operands);
  return success();
}

mlir::LogicalResult mlir::spn::BodyLowering::matchAndRewrite(mlir::spn::low::SPNBody op,
                                                             OpAdaptor adaptor,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
  assert(operands.size() == op.getBody().front().getNumArguments() &&
      "Expecting the number of operands to match the block arguments");

  SmallVector<Value> argValues;
  for (auto opArg : llvm::zip(operands, op.getBody().front().getArguments())) {
    auto operand = std::get<0>(opArg);
    auto arg = std::get<1>(opArg);
    if (arg.getType().isa<low::LogType>()) {
      auto convertLog = rewriter.create<low::SPNConvertLog>(op->getLoc(), operand);
      if (auto vectorOp = dyn_cast<low::LoSPNVectorizable>(operand.getDefiningOp())) {
        convertLog.setVectorized(vectorOp.getVectorWidth());
      }
      argValues.push_back(convertLog);
    } else {
      argValues.push_back(operand);
    }
  }

  SmallVector<Value, 2> resultValues;
  op.getBody().front().walk([&](low::SPNYield yield) {
    for (auto res : yield.getResultValues()) {
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
  rewriter.mergeBlockBefore(&op.getBody().front(), op, argValues);
  rewriter.replaceOp(op, resultValues);
  return success();
}
