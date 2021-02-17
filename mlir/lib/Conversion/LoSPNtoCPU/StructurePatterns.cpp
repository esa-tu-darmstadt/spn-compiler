//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <mlir/IR/BlockAndValueMapping.h>
#include "LoSPNtoCPU/StructurePatterns.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"

mlir::LogicalResult mlir::spn::KernelLowering::matchAndRewrite(mlir::spn::low::SPNKernel op,
                                                               llvm::ArrayRef<mlir::Value> operands,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.empty() && "Kernel should not take any operands");
  auto replaceFunc = rewriter.create<mlir::FuncOp>(op.getLoc(), op.kernelName(), op.getType());
  auto funcBlock = replaceFunc.addEntryBlock();
  rewriter.mergeBlocks(&op.body().front(), funcBlock, funcBlock->getArguments());
  rewriter.eraseOp(op);
  return success();
}

mlir::LogicalResult mlir::spn::BatchTaskLowering::matchAndRewrite(mlir::spn::low::SPNTask op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
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
  auto ub = rewriter.create<DimOp>(op.getLoc(), taskBlock->getArgument(0), 0);
  auto step = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1));
  auto loop = rewriter.create<scf::ForOp>(op.getLoc(), const0, ub, step);
  rewriter.create<ReturnOp>(op->getLoc());
  // Fill the loop
  auto loopBlock = rewriter.createBlock(&loop.getLoopBody());
  rewriter.setInsertionPointToStart(loopBlock);
  BlockAndValueMapping batchRead;
  op.body().front().walk([&](low::SPNBatchRead read) {
    auto constIndex = rewriter.create<ConstantOp>(op->getLoc(), rewriter.getIndexAttr(read.sampleIndex()));
    auto load = rewriter.create<LoadOp>(op.getLoc(), read.batchMem(),
                                        ValueRange{loop.getInductionVar(), constIndex});
    batchRead.map(read, load);
  });
  BlockAndValueMapping results;
  op.body().front().walk([&](low::SPNBody body) {
    assert(body->getOperands().size() == body.body().front().getNumArguments());
    BlockAndValueMapping bodyArgs;
    for (auto opArg : llvm::zip(body.getOperands(), body.body().front().getArguments())) {
      bodyArgs.map(std::get<1>(opArg), batchRead.lookup(std::get<0>(opArg)));
    }
    for (auto& operation : body.body().front()) {
      if (auto yield = dyn_cast<low::SPNYield>(operation)) {
        for (auto resVal : llvm::zip(body.getResults(), yield.getOperands())) {
          std::get<1>(resVal).dump();
          bodyArgs.lookup(std::get<1>(resVal)).dump();
          results.map(std::get<0>(resVal),
                      bodyArgs.lookup(std::get<1>(resVal)));
        }
        continue;
      }
      rewriter.clone(operation, bodyArgs);
    }
    rewriter.eraseOp(body);
  });
  op.body().front().walk([&](low::SPNBatchWrite write) {
    // TODO Handle case of multiple results written to same memref.
    auto resVal = results.lookup(write.results().front());
    resVal.getDefiningOp()->dump();
    rewriter.create<StoreOp>(op.getLoc(), resVal, write.batchMem(), loop.getInductionVar());
    rewriter.eraseOp(write);
  });
  // Insert a call to the newly created task function.
  rewriter.restoreInsertionPoint(restore);
  rewriter.replaceOpWithNewOp<mlir::CallOp>(op, taskFunc, operands);
  op->getParentOfType<ModuleOp>().dump();
  llvm::dbgs() << "Uses empty? " << op->getUses().empty() << "\n";
  for (auto& operation : op.body().front()) {
    llvm::dbgs() << "Uses empty? " << operation.getName() << ": " << operation.getUses().empty() << "\n";
    if (!operation.getUses().empty()) {
      for (auto U : operation.getUsers()) {
        llvm::dbgs() << U->getName() << "\n";
      }
    }
  }
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
      resultValues.push_back(res);
    }
    rewriter.eraseOp(yield);
  });
  rewriter.mergeBlockBefore(&op.body().front(), op, operands);
  rewriter.replaceOp(op, resultValues);
  op->getParentOfType<ModuleOp>().dump();
  llvm::dbgs() << "Uses empty? " << op->getUses().empty() << "\n";
  for (auto U : op->getUsers()) {
    U->dump();
  }
  return success();
}
