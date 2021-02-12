//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNBufferizationPatterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

mlir::LogicalResult mlir::spn::low::TaskBufferize::matchAndRewrite(mlir::spn::low::SPNTask op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  assert(!operands.empty() && "Expecting at least one input to a task");
  assert(operands[0].getType().isa<MemRefType>());
  // All inputs to this Task will also become inputs to the newly create task.
  SmallVector<Value, 10> inputs;
  for (auto operand : operands) {
    inputs.push_back(operand);
  }
  // Retrieve the dynamic dimension of the first MemRef input, which corresponds
  // to the actual number of samples in the batch.
  // This value will be used for the allocation for the results of this task and
  // assumes that all inputs/outputs use the same number of samples.
  auto batchDim = rewriter.create<mlir::DimOp>(op.getLoc(), operands[0], 0);
  SmallVector<Value, 1> dynSizes;
  dynSizes.push_back(batchDim);
  // Create a MemRef via allocation for each result. Instead of returning a Tensor, the task
  // will store its results into these allocated memories after bufferization.
  // The allocated MemRefs will also become inputs to this Task.
  SmallVector<Value, 2> allocations;
  for (auto r : op->getResults()) {
    assert(r.getType().isa<TensorType>());
    auto memRefType = typeConverter->convertType(r.getType());
    auto alloc = rewriter.create<mlir::AllocOp>(op.getLoc(), memRefType,
                                                dynSizes, ValueRange{}, IntegerAttr());
    inputs.push_back(alloc);
    allocations.push_back(alloc);
  }
  // Create a new SPNTask, with the original inputs + the allocated memories as input.
  auto newTask = rewriter.create<mlir::spn::low::SPNTask>(op->getLoc(), TypeRange{}, inputs, op.batchSize());
  // Create a block with block arguments.
  auto newTaskBlock = rewriter.createBlock(&newTask.body());
  SmallVector<Value, 2> inArgs;
  SmallVector<Value, 2> outArgs;
  for (auto arg : llvm::enumerate(inputs)) {
    auto blockArg = newTaskBlock->addArgument(arg.value().getType());
    if (arg.index() < operands.size()) {
      inArgs.push_back(blockArg);
    } else {
      outArgs.push_back(blockArg);
    }
  }
  // Merge the body of the original SPNTask into the new Task.
  rewriter.mergeBlocks(&op.body().front(), newTaskBlock, inArgs);
  // A Task before bufferization should return Tensors and should be terminated
  // by a SPNBatchCollect. Create a SPNBatchWrite for each result in the SPNBatchCollect
  // (assumes a 1:1 mapping between scalar operand of SPNBatchCollect and Tensor result).
  // Insert a SPNReturn (with no return values) as new terminator for the new task.
  newTaskBlock->walk([&](low::SPNBatchCollect collect) {
    SmallVector<Value, 2> scalarReturnValues;
    for (auto retVal : llvm::zip(collect.results(), collect.tensors(), outArgs)) {
      Value scalarResult = std::get<0>(retVal);
      auto convertedType = typeConverter->convertType(std::get<1>(retVal).getType());
      auto memRef = std::get<2>(retVal);
      assert(convertedType == memRef.getType());
      rewriter.create<low::SPNBatchWrite>(collect.getLoc(), scalarResult, memRef);
    }
    rewriter.create<low::SPNReturn>(collect->getLoc(), ValueRange{});
    rewriter.eraseOp(collect);
  });
  // The results computed by the new task are stored in the allocated MemRefs,
  // so users of the original Task should use those after bufferization.
  rewriter.replaceOp(op, allocations);
  return success();
}

mlir::LogicalResult mlir::spn::low::BatchExtractBufferize::matchAndRewrite(mlir::spn::low::SPNBatchExtract op,
                                                                           llvm::ArrayRef<mlir::Value> operands,
                                                                           mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands[0].getType().isa<MemRefType>());
  rewriter.replaceOpWithNewOp<low::SPNBatchRead>(op, op.getType(), operands[0], op.sampleIndex());
  return success();
}
