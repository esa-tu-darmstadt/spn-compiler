//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <mlir/IR/BlockAndValueMapping.h>
#include "LoSPNtoCPU/Vectorization/VectorizationPatterns.h"
#include "../Target/TargetInformation.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"

mlir::LogicalResult mlir::spn::VectorizeBatchTask::matchAndRewrite(mlir::spn::low::SPNTask op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  static int taskCount = 0;

  if (op.batchSize() <= 1) {
    return rewriter.notifyMatchFailure(op,
                                       "Specialized for batch vectorization, does not match for batchSize == 1");
  }

  assert(operands.back().getType().isa<MemRefType>());
  auto computationType = operands.back().getType().dyn_cast<MemRefType>().getElementType();
  auto hwVectorWidth = TargetInformation::nativeCPUTarget().getHWVectorEntries(computationType);

  if (hwVectorWidth <= 1) {
    return rewriter.notifyMatchFailure(op,
                                       llvm::formatv(
                                           "No vectorization possible for data-type {} on the requested target",
                                           computationType));
  }

  // Check if all nodes can be vectorized before trying to do so.
  auto allVectorizable = op.body().walk([hwVectorWidth](low::LoSPNVectorizable vOp) {
    if (!vOp.isVectorizable(hwVectorWidth)) {
      vOp.emitRemark() << "Operation cannot be vectorized with vector width " << hwVectorWidth;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (allVectorizable.wasInterrupted()) {
    return rewriter.notifyMatchFailure(op, "Not all nested operations can be vectorized, aborting vectorization");
  }

  // Let the user know which vector width will be used.
  op->emitRemark() << "Attempting to vectorize with vector width " << hwVectorWidth
                   << " for data-type " << computationType;

  // Emit a warning if the target vector width does not divide the requested batch size.
  // This will cause a part of each batch (batchSize % vectorWidth elements) to be processed
  // by the scalar epilog loop instead of the vectorized loop.
  if ((op.batchSize() % hwVectorWidth) != 0) {
    op.emitWarning() << "The target vector width " << hwVectorWidth
                     << " does not divide the requested batch size " << op.batchSize()
                     << "; This can result in degraded performance. "
                     << "Choose the batch size as a multiple of the vector width "
                     << hwVectorWidth;
  }

  auto restore = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(op->getParentOfType<mlir::ModuleOp>().getBody());
  SmallVector<Type, 5> inputTypes;
  for (auto operand : operands) {
    inputTypes.push_back(operand.getType());
  }
  auto funcType = FunctionType::get(rewriter.getContext(), inputTypes, {});
  auto taskFunc = rewriter.create<FuncOp>(op->getLoc(), Twine("vec_task_", std::to_string(taskCount++)).str(),
                                          funcType);
  auto taskBlock = taskFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(taskBlock);
  auto numSamples = rewriter.create<DimOp>(op.getLoc(), taskBlock->getArgument(0), 0);
  auto vectorWidthConst = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(hwVectorWidth));
  auto remainder = rewriter.create<mlir::UnsignedRemIOp>(op.getLoc(), numSamples, vectorWidthConst);
  auto ubVectorized = rewriter.create<mlir::SubIOp>(op.getLoc(), numSamples, remainder);

  // Create the vectorized loop, iterating from 0 to ubVectorized, in steps of hwVectorWidth.
  auto lbVectorized = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  auto stepVectorized = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(hwVectorWidth));
  auto vectorizedLoop = rewriter.create<mlir::scf::ForOp>(op.getLoc(), lbVectorized, ubVectorized, stepVectorized);
  auto& vectorLoopBody = vectorizedLoop.getLoopBody().front();

  auto restoreTask = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&vectorLoopBody);
  auto oldTaskArgs = op.body().front().getArguments();
  BlockAndValueMapping mapVectorTaskArgs;
  // Map from batchIndex to vectorized loop induction var.
  mapVectorTaskArgs.map(oldTaskArgs.front(), vectorizedLoop.getInductionVar());
  int i = 1;
  for (auto bArg : taskBlock->getArguments()) {
    mapVectorTaskArgs.map(oldTaskArgs[i++], bArg);
  }
  // Copy the operations from the Task's content to the vectorized loop
  for (auto& node : op.body().front()) {
    if (isa<low::SPNReturn>(&node)) {
      continue;
    }
    (void) rewriter.clone(node, mapVectorTaskArgs);
  }

  // Mark all operations contained in the vectorized loop as vectorized.
  vectorLoopBody.walk([hwVectorWidth](low::LoSPNVectorizable vOp) {
    vOp.setVectorized(hwVectorWidth);
  });

  rewriter.restoreInsertionPoint(restoreTask);

  // Create the scalar epilog loop, iterating from ubVectorized to numSamples, in steps of 1.
  auto stepScalar = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1));
  auto scalarLoop = rewriter.create<mlir::scf::ForOp>(op.getLoc(), ubVectorized, numSamples, stepScalar);
  auto& scalarLoopBody = scalarLoop.getLoopBody().front();

  restoreTask = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&scalarLoopBody);
  SmallVector<Value, 5> blockReplacementArgs;
  blockReplacementArgs.push_back(scalarLoop.getInductionVar());
  for (auto bArg : taskBlock->getArguments()) {
    blockReplacementArgs.push_back(bArg);
  }
  rewriter.mergeBlockBefore(&op.body().front(), scalarLoopBody.getTerminator(), blockReplacementArgs);
  scalarLoopBody.walk([&rewriter](low::SPNReturn ret) {
    assert(ret.returnValues().empty() && "Task return should be empty");
    rewriter.eraseOp(ret);
  });

  rewriter.restoreInsertionPoint(restoreTask);
  rewriter.create<ReturnOp>(op->getLoc());
  // Insert a call to the newly created task function.
  rewriter.restoreInsertionPoint(restore);
  rewriter.replaceOpWithNewOp<mlir::CallOp>(op, taskFunc, operands);
  return success();

}
