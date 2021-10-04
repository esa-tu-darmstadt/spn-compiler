//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoGPU/GPUStructurePatterns.h"
#include <mlir/IR/BlockAndValueMapping.h>
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

mlir::LogicalResult mlir::spn::KernelGPULowering::matchAndRewrite(mlir::spn::low::SPNKernel op,
                                                                  llvm::ArrayRef <mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.empty() && "Kernel should not take any operands");
  auto replaceFunc = rewriter.create<mlir::FuncOp>(op.getLoc(), op.getName(), op.getType());
  auto funcBlock = replaceFunc.addEntryBlock();
  rewriter.mergeBlocks(&op.body().front(), funcBlock, funcBlock->getArguments());
  rewriter.eraseOp(op);
  return success();
}

mlir::LogicalResult mlir::spn::BatchTaskGPULowering::matchAndRewrite(mlir::spn::low::SPNTask op,
                                                                     llvm::ArrayRef<mlir::Value> operands,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
  //
  // Lower a task with batchSize > 1. The task is lowered to a function, containing a scalar loop iterating over the
  // samples in the batch. The content of the Task is merged into the newly created loop's body, the loop induction
  // variable replaces the batchIndex argument of the Task.
  if (op.batchSize() == 1) {
    return rewriter.notifyMatchFailure(op, "Match only batched (batchSize > 1) execution");
  }
  //
  // Allocate device memory for the inputs and out-args of the task.
  //
  SmallVector<std::pair<Value, Value>, 2> copyTo;
  SmallVector<std::pair<Value, Value>, 2> copyFrom;
  SmallVector<Value> deviceMemories;
  auto& bodyBlock = op.body().front();
  assert(operands.size() + 1 == bodyBlock.getNumArguments());
  for (unsigned i = 0; i < operands.size(); ++i) {
    auto taskInput = operands[i];
    // Skip the first block argument, i.e. the batch index.
    auto blockArg = bodyBlock.getArgument(i + 1);
    assert(blockArg.getType().isa<MemRefType>());
    bool isRead = false;
    bool isWritten = false;
    for (auto U : blockArg.getUsers()) {
      if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(U)) {
        SmallVector<MemoryEffects::EffectInstance, 1> effects;
        memEffect.getEffectsOnValue(blockArg, effects);
        for (auto e : effects) {
          if (isa<MemoryEffects::Read>(e.getEffect())) {
            isRead = true;
          }
          if (isa<MemoryEffects::Write>(e.getEffect())) {
            isWritten = true;
          }
        }
      } else {
        // Pessimistically assume both read and written.
        isRead = true;
        isWritten = true;
      }
    }
    auto inputType = taskInput.getType().dyn_cast<MemRefType>();
    assert(inputType && inputType.hasRank());
    SmallVector<Value, 1> dynamicSizes;
    for (int k = 0; k < inputType.getRank(); ++k) {
      if (inputType.isDynamicDim(k)) {
        auto dymSize = rewriter.create<memref::DimOp>(op.getLoc(), taskInput, k);
        dynamicSizes.push_back(dymSize);
      }
    }
    auto memref = MemRefType::get(inputType.getShape(), inputType.getElementType());
    auto deviceMem = rewriter.create<gpu::AllocOp>(op->getLoc(), memref, ValueRange{}, dynamicSizes, ValueRange{});
    deviceMemories.push_back(deviceMem.memref());
    if (isRead) {
      copyTo.emplace_back(taskInput, deviceMem.memref());
    }
    if (isWritten) {
      copyFrom.emplace_back(deviceMem.memref(), taskInput);
    }

  }


  //
  // Copy all memrefs that are read from in the GPU task from host memory to device memory.
  //
  for (auto& hostDevice : copyTo) {
    auto hostMem = hostDevice.first;
    auto deviceMem = hostDevice.second;
    rewriter.create<gpu::MemcpyOp>(op->getLoc(), llvm::None, ValueRange{}, deviceMem, hostMem);
  }

  // Determine the total number of samples to compute from the size of the memref passed as the
  // first argument to the Task.
  auto inputMemRefTy = operands[0].getType().dyn_cast<MemRefType>();
  assert(inputMemRefTy);
  assert(inputMemRefTy.hasRank() && inputMemRefTy.getRank() == 2);
  assert(inputMemRefTy.isDynamicDim(0) ^ inputMemRefTy.isDynamicDim(1));
  auto index = (inputMemRefTy.isDynamicDim(0)) ? 0 : 1;
  auto numSamples = rewriter.create<memref::DimOp>(op.getLoc(), operands[0], index);
  // We assume 1D layout of threads and blocks and use the user-specified batchSize as blockSize.
  if ((op.batchSize() % 32) != 0) {
    op.emitWarning() << "Batch size should be a multiple of the warp-size (32)";
  }
  auto blockSize = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(op.batchSize()));
  auto numBlocks = rewriter.create<SignedCeilDivIOp>(op.getLoc(), numSamples, blockSize);
  auto constantOne = rewriter.create<ConstantOp>(op->getLoc(), rewriter.getIndexAttr(1));
  auto gpuLaunch = rewriter.create<gpu::LaunchOp>(op->getLoc(), numBlocks, constantOne, constantOne,
                                                  blockSize, constantOne, constantOne);
  auto restore = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&gpuLaunch.body().front());
  auto blockOffset = rewriter.create<MulIOp>(op->getLoc(), gpuLaunch.blockSizeX(), gpuLaunch.getBlockIds().x);
  auto batchIndex = rewriter.create<AddIOp>(op.getLoc(), blockOffset, gpuLaunch.getThreadIds().x);
  SmallVector<Value, 5> blockReplacementArgs;
  blockReplacementArgs.push_back(batchIndex);
  for (auto mem : deviceMemories) {
    // Use the previously allocated GPU device memory.
    blockReplacementArgs.push_back(mem);
  }
  auto checkInBounds = rewriter.create<mlir::CmpIOp>(op->getLoc(), CmpIPredicate::ult, batchIndex, numSamples);
  auto ifOp = rewriter.create<mlir::scf::IfOp>(op.getLoc(), checkInBounds, false);
  (void) rewriter.create<gpu::TerminatorOp>(op.getLoc());
  rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
  rewriter.mergeBlockBefore(&op.body().front(), ifOp.thenRegion().front().getTerminator(), blockReplacementArgs);
  gpuLaunch.body().front().walk([&rewriter](low::SPNReturn ret) {
    assert(ret.returnValues().empty() && "Task return should be empty");
    rewriter.eraseOp(ret);
  });
  rewriter.restoreInsertionPoint(restore);
  //
  // Copy back all device memories that the GPU task writes to
  //
  for (auto& deviceHost : copyFrom) {
    auto deviceMem = deviceHost.first;
    auto hostMem = deviceHost.second;
    rewriter.create<gpu::MemcpyOp>(op->getLoc(), llvm::None, ValueRange{}, hostMem, deviceMem);
  }
  // Deallocation is not performed here, because the buffer might be re-used by other tasks on the GPU to
  // eliminate unnecessary copies between host and device for intermediate results.
  // Therefore, deallocations are later on inserted by a dedicated pass.
  rewriter.eraseOp(op);
  return success();
}

mlir::LogicalResult mlir::spn::BodyGPULowering::matchAndRewrite(mlir::spn::low::SPNBody op,
                                                                llvm::ArrayRef<mlir::Value> operands,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.size() == op.body().front().getNumArguments() &&
      "Expecting the number of operands to match the block arguments");

  SmallVector<Value> argValues;
  for (auto opArg : llvm::zip(operands, op.body().front().getArguments())) {
    auto operand = std::get<0>(opArg);
    auto arg = std::get<1>(opArg);
    if (arg.getType().isa<low::LogType>()) {
      auto convertLog = rewriter.create<low::SPNConvertLog>(op->getLoc(), operand);
      argValues.push_back(convertLog);
    } else {
      argValues.push_back(operand);
    }
  }

  SmallVector<Value, 2> resultValues;
  op.body().front().walk([&](low::SPNYield yield) {
    for (auto res : yield.resultValues()) {
      if (auto logType = res.getType().dyn_cast<low::LogType>()) {
        // If the body internally computes in log-space, we need to
        // strip the log-semantic for the operations using the result of the body.
        rewriter.setInsertionPoint(yield);
        auto stripLog = rewriter.create<low::SPNStripLog>(yield->getLoc(), res, logType.getBaseType());
        resultValues.push_back(stripLog);
      } else {
        resultValues.push_back(res);
      }
    }
    rewriter.eraseOp(yield);
  });
  rewriter.mergeBlockBefore(&op.body().front(), op, argValues);
  rewriter.replaceOp(op, resultValues);
  return success();
}
