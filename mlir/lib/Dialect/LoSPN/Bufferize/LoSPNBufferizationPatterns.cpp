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
  // Insert a first block argument corresponding to the batch index.
  auto batchIndex = newTaskBlock->addArgument(rewriter.getIndexType());
  inArgs.push_back(batchIndex);
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
      rewriter.create<low::SPNBatchWrite>(collect.getLoc(), scalarResult, memRef, batchIndex);
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
  assert(operands[1].getType().isa<IndexType>());
  rewriter.replaceOpWithNewOp<low::SPNBatchRead>(op, op.getType(), operands[0],
                                                 operands[1], op.sampleIndex());
  return success();
}

mlir::LogicalResult mlir::spn::low::KernelBufferize::matchAndRewrite(mlir::spn::low::SPNKernel op,
                                                                     llvm::ArrayRef<mlir::Value> operands,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
  //
  // Bufferize an SPNKernel. The bufferization does not only convert the
  // types of the inputs & outputs, but also transforms all outputs into
  // out-args, i.e., the caller needs to pass in a buffer and the SPNKernel
  // and its respective sub-tasks store the result into these buffers.
  assert(operands.empty() && "SPNKernel should not receive any operands");
  SmallVector<Type> newInputTypes;
  unsigned numInputs = 0;
  // Convert the input and output types.
  for (auto inTy : op.getType().getInputs()) {
    newInputTypes.push_back(typeConverter->convertType(inTy));
    ++numInputs;
  }
  for (auto outTy : op.getType().getResults()) {
    newInputTypes.push_back(typeConverter->convertType(outTy));
  }
  // Construct a new kernel with a fucntion type that does produce any results, but
  // has the same inputs and additional out-args for all results.
  auto newKernelType = FunctionType::get(rewriter.getContext(), newInputTypes, TypeRange{});
  auto newKernel = rewriter.create<low::SPNKernel>(op->getLoc(), op.getName(), newKernelType);
  auto newKernelBlock = newKernel.addEntryBlock();
  rewriter.setInsertionPointToStart(newKernelBlock);
  SmallVector<Value, 5> inArgs;
  SmallVector<Value, 5> outArgs;
  unsigned count = 0;
  // Distinguish between the original input arguments and the
  // newly introduced out-args.
  for (auto arg : newKernelBlock->getArguments()) {
    if (count < numInputs) {
      inArgs.push_back(arg);
    } else {
      outArgs.push_back(arg);
    }
    ++count;
  }
  // Merge the block of the original Kernel into the new one's body.
  rewriter.mergeBlocks(&op.body().front(), newKernelBlock, inArgs);
  // Walk the returns of the new body and insert copy from the result value
  // to the newly created out-args.
  newKernelBlock->walk([&](SPNReturn ret) {
    SmallVector<Value, 2> scalarReturns;
    unsigned count = 0;
    for (auto retVal : ret->getOperands()) {
      if (!typeConverter->isLegal(retVal.getType())) {
        retVal = typeConverter->materializeTargetConversion(rewriter, ret.getLoc(),
                                                            typeConverter->convertType(retVal.getType()),
                                                            retVal);
      }
      if (retVal.getType().isa<MemRefType>()) {
        rewriter.create<low::SPNCopy>(ret.getLoc(), retVal, outArgs[count++]);
      } else {
        scalarReturns.push_back(retVal);
      }
    }
    rewriter.create<low::SPNReturn>(op->getLoc(), scalarReturns);
    rewriter.eraseOp(ret);
  });
  // Delete the old SPNKernel.
  rewriter.eraseOp(op);
  return success();
}
