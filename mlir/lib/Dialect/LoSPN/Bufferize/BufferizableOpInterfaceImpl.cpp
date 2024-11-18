#include "LoSPN/Bufferize/BufferizableOpInterfaceImpl.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::spn::low;
using namespace mlir::bufferization;

namespace {
static SPNReturn getAssumedUniqueReturnOp(SPNTask taskOp) {
  SPNReturn returnOp;
  for (Block &b : taskOp.getBody()) {
    if (auto candidateOp = dyn_cast<SPNReturn>(b.getTerminator())) {
      if (returnOp)
        return nullptr;
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

static SPNReturn getAssumedUniqueReturnOp(SPNKernel kernelOp) {
  SPNReturn returnOp;
  for (Block &b : kernelOp.getBody()) {
    if (auto candidateOp = dyn_cast<SPNReturn>(b.getTerminator())) {
      if (returnOp)
        return nullptr;
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

/// All tensors are converted to memrefs with static identity layout.
static BaseMemRefType convertTensorType(TensorType tensorType,
                                        const BufferizationOptions &options) {
  return bufferization::getMemRefTypeWithStaticIdentityLayout(
      tensorType, *options.defaultMemorySpaceFn(tensorType));
}

/// Bufferization of SPNBatchExtract. Replace with SPNBatchRead.
struct BatchExtractInterface
    : public BufferizableOpInterface::ExternalModel<BatchExtractInterface,
                                                    SPNBatchExtract> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getOpResult(0) /*result*/, BufferRelation::Equivalent,
             /*isDefinite=*/false}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto batchExtractOp = cast<SPNBatchExtract>(op);

    FailureOr<Value> maybeInputBuffer =
        getBuffer(rewriter, batchExtractOp.getInput(), options);
    if (failed(maybeInputBuffer))
      return failure();

    Value inputBuffer = *maybeInputBuffer;

    replaceOpWithNewBufferizedOp<SPNBatchRead>(
        rewriter, batchExtractOp, inputBuffer, batchExtractOp.getDynamicIndex(),
        batchExtractOp.getStaticIndex(), batchExtractOp.getTransposed());
    return success();
  }
};

/// Bufferization of SPNBatchCollect is done as part of the SPNReturn
/// bufferization.
struct BatchCollectInterface
    : public BufferizableOpInterface::ExternalModel<BatchCollectInterface,
                                                    SPNBatchCollect> {

  bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

  /// Return the buffer type for the value owned by the given task, which is
  /// either a block argument or the result of the task.
  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto tensorType = dyn_cast<TensorType>(value.getType());
    assert(tensorType && "expected TensorType");

    return convertTensorType(tensorType, options);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return success();
  }
};

/// Bufferization of SPNReturn.
/// For tasks, add an memref out-arg to the block, replace the batch collects
/// with batch writes to the out-arg, and return nothing.
/// For kernels, add an memref out-arg to the block, add a copy from the batch

struct ReturnOpInterface
    : public BufferizableOpInterface::ExternalModel<ReturnOpInterface,
                                                    SPNReturn> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    Operation *parentOp = op->getParentOp();
    SPNReturn returnOp = cast<SPNReturn>(op);
    if (isa<SPNKernel>(parentOp))
      return bufferizeReturnInKernel(returnOp, rewriter, options);

    assert(isa<SPNTask>(parentOp) && "expected SPNTask");
    return bufferizeReturnInTask(returnOp, rewriter, options);
  }

  LogicalResult
  bufferizeReturnInTask(SPNReturn returnOp, RewriterBase &rewriter,
                        const BufferizationOptions &options) const {

    Operation *parentOp = returnOp->getParentOp();
    SPNTask taskOp = cast<SPNTask>(parentOp);

    for (Value operand : returnOp.getReturnValues()) {
      // We expect the return values to be the results of SPNBatchCollect ops.
      auto batchCollectOp = dyn_cast<SPNBatchCollect>(operand.getDefiningOp());
      if (!batchCollectOp)
        return returnOp->emitError(
            "expected return values to be the results of SPNBatchCollect ops");

      // Add an out-arg for the result.
      FailureOr<BaseMemRefType> maybeBufferType =
          bufferization::getBufferType(operand, options);
      if (failed(maybeBufferType))
        return failure();
      BlockArgument outArg =
          taskOp.getBody().addArgument(*maybeBufferType, returnOp.getLoc());

      // Replace the batch collect op with a batch write to the out-arg.
      rewriter.setInsertionPoint(batchCollectOp);
      rewriter.create<SPNBatchWrite>(
          returnOp->getLoc(), outArg, batchCollectOp.getBatchIndex(),
          batchCollectOp.getResultValues(), batchCollectOp.getTransposedAttr());
      bufferization::replaceOpWithBufferizedValues(rewriter, batchCollectOp,
                                                   outArg);
    }

    // Remove the operands from the return op.
    returnOp.getReturnValuesMutable().clear();

    return success();
  }

  LogicalResult
  bufferizeReturnInKernel(SPNReturn returnOp, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {

    Operation *parentOp = returnOp->getParentOp();
    SPNKernel kernelOp = cast<SPNKernel>(parentOp);

    for (Value operand : returnOp.getReturnValues()) {
      // Add an out-arg for the result.
      FailureOr<BaseMemRefType> maybeBufferType =
          bufferization::getBufferType(operand, options);
      if (failed(maybeBufferType))
        return failure();
      BlockArgument outArg =
          kernelOp.getBody().addArgument(*maybeBufferType, returnOp.getLoc());

      // Add a copy from the bufferized return value to the out-arg.
      rewriter.setInsertionPoint(returnOp);
      FailureOr<Value> maybeReturnValue =
          bufferization::getBuffer(rewriter, operand, options);
      if (failed(maybeReturnValue))
        return failure();
      rewriter.create<memref::CopyOp>(returnOp.getLoc(),
                                      maybeReturnValue.value(), outArg);
    }

    // Remove the operands from the return op.
    returnOp.getReturnValuesMutable().clear();

    return success();
  }
};

/// Bufferization of SPNTask. Bufferize the operands, allocate buffers for the
/// results, add the results as out args, and replace the return op with copies
/// to the out args.
struct TaskInterface
    : public BufferizableOpInterface::ExternalModel<TaskInterface, SPNTask> {

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  bool hasTensorSemantics(Operation *op) const {
    auto isaTensor = llvm::IsaPred<TensorType>;

    // A task has tensor semantics if it has tensor arguments/results.
    auto taskOp = cast<SPNTask>(op);
    bool hasTensorArg = any_of(taskOp->getOperandTypes(), isaTensor);
    bool hasTensorResult = any_of(taskOp.getResultTypes(), isaTensor);
    if (hasTensorArg || hasTensorResult)
      return true;

    return false;
  }

  AliasingOpOperandList
  getAliasingOpOperands(Operation *op, Value value,
                        const AnalysisState &state) const {
    return {}; // FIXME
    // return getAliasingBranchOpOperands(op, cast<BlockArgument>(value),
    // state);
  }

  /// Return the buffer type for the value owned by the given task, which is
  /// either a block argument or the result of the task.
  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto tensorType = dyn_cast<TensorType>(value.getType());
    assert(tensorType && "expected TensorType");

    return convertTensorType(tensorType, options);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto taskOp = cast<SPNTask>(op);
    auto kernelOp = cast<SPNKernel>(taskOp->getParentOp());
    assert(kernelOp && "expected task to be a child of a kernel");

    // Bufferize the body block.
    assert(taskOp.getBody().getBlocks().size() == 1 &&
           "tasks are expected to have a single block");
    if (failed(bufferization::bufferizeBlockSignature(&taskOp.getBody().front(),
                                                      rewriter, options)))
      return failure();

    // Bufferize the task's operands
    SmallVector<Value> newOperands;
    for (OpOperand &operand : taskOp->getOpOperands()) {
      if (!operand.get().getType().isa<TensorType>()) {
        newOperands.push_back(operand.get());
        continue;
      }
      FailureOr<Value> bufferizedOperand =
          bufferization::getBuffer(rewriter, operand.get(), options);
      if (failed(bufferizedOperand))
        return failure();

      newOperands.push_back(bufferizedOperand.value());
    }

    // Get the size of the dynamic dimension. Required so that we can later
    // allocate buffers for the results.
    BlockArgument kernelInputArg = kernelOp.getBody().getArgument(0);
    FailureOr<Value> maybeKernelInputArgBuffer =
        bufferization::getBuffer(rewriter, kernelInputArg, options);
    if (failed(maybeKernelInputArgBuffer))
      return failure();
    Value kernelInputArgBuffer = *maybeKernelInputArgBuffer;
    Value batchSize = rewriter.create<memref::DimOp>(
        taskOp.getLoc(), kernelInputArgBuffer, /*index=*/0);

    // Allocate buffers for the results.
    SmallVector<Value> bufferizedResults;
    rewriter.setInsertionPoint(taskOp);
    for (OpResult result : taskOp->getResults()) {

      FailureOr<BaseMemRefType> maybeBufferType =
          bufferization::getBufferType(result, options);
      if (failed(maybeBufferType))
        return failure();

      Value buffer = rewriter.create<memref::AllocaOp>(
          taskOp.getLoc(), *maybeBufferType, batchSize, ValueRange{},
          IntegerAttr());
      newOperands.push_back(buffer);
      bufferizedResults.push_back(buffer);
    }

    // Create a new task with the bufferized operands and without the return
    // values.
    auto newTask = rewriter.create<SPNTask>(
        taskOp.getLoc(), TypeRange{}, newOperands, taskOp.getBatchSizeAttr());

    // Move the body of the old task to the new task.
    rewriter.moveBlockBefore(&taskOp.getBody().front(), &newTask.getBody(),
                             newTask.getBody().end());

    //  Replace the task with the bufferized results.
    bufferization::replaceOpWithBufferizedValues(rewriter, taskOp,
                                                 bufferizedResults);

    return success();
  }

  /// Return `true` if the given function argument is writable.
  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    // All function arguments are writable by default.
    return true;
  }
};

/// Bufferization of a kernel
struct KernelInterface
    : public BufferizableOpInterface::ExternalModel<KernelInterface,
                                                    SPNKernel> {
  /// Check if the given kernel has tensor semantics.
  bool hasTensorSemantics(Operation *op) const {
    auto kernelOp = cast<SPNKernel>(op);
    auto isaTensor = llvm::IsaPred<TensorType>;

    // A kernel has tensor semantics if its function type has tensor arguments
    // or results.
    FunctionType funcType = kernelOp.getFunctionType();

    bool hasTensorArg = any_of(funcType.getInputs(), isaTensor);
    bool hasTensorResult = any_of(funcType.getResults(), isaTensor);
    if (hasTensorArg || hasTensorResult)
      return true;

    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto kernelOp = cast<SPNKernel>(op);
    Block *kernelBlock = &kernelOp.getBody().front();
    // Bufferize the body block.
    assert(kernelOp.getBody().getBlocks().size() == 1 &&
           "tasks are expected to have a single block");
    if (failed(bufferization::bufferizeBlockSignature(kernelBlock, rewriter,
                                                      options)))
      return failure();

    // Collect the bufferized input and out-arg types
    SmallVector<Type> bufferizedArgTypes;
    for (Type inputType : kernelOp.getFunctionType().getInputs()) {
      if (auto tensorType = dyn_cast<TensorType>(inputType)) {
        bufferizedArgTypes.push_back(convertTensorType(tensorType, options));
        continue;
      }
      bufferizedArgTypes.push_back(inputType);
    }
    for (Type resultType : kernelOp.getFunctionType().getResults()) {
      if (auto tensorType = dyn_cast<TensorType>(resultType)) {
        bufferizedArgTypes.push_back(convertTensorType(tensorType, options));
        continue;
      }
      bufferizedArgTypes.push_back(resultType);
    }

    // Modify the function type of the kernel.
    FunctionType newFuncType = FunctionType::get(
        kernelOp.getContext(), bufferizedArgTypes, {} /*results*/);
    kernelOp.setFunctionType(newFuncType);

    return success();
  }

  /// Returns the buffer type for the given block argument.
  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto tensorType = dyn_cast<TensorType>(value.getType());
    assert(tensorType && "expected TensorType");

    return convertTensorType(tensorType, options);
  }
};

/// Bufferization of SPNGather.
struct GatherInterface
    : public BufferizableOpInterface::ExternalModel<GatherInterface,
                                                    SPNGather> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto gatherOp = cast<SPNGather>(op);

    FailureOr<BaseMemRefType> maybeResultType =
        bufferization::getBufferType(gatherOp.getResult(), options);
    if (failed(maybeResultType))
      return failure();

    FailureOr<Value> maybeInputBuffer =
        bufferization::getBuffer(rewriter, gatherOp.getInput(), options);
    if (failed(maybeInputBuffer))
      return failure();

    bufferization::replaceOpWithNewBufferizedOp<SPNGather>(
        rewriter, op, *maybeResultType, *maybeInputBuffer,
        gatherOp.getIndices());

    return success();
  }

  /// Returns the buffer type for the given block argument.
  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto tensorType = dyn_cast<TensorType>(value.getType());
    assert(tensorType && "expected TensorType");

    return convertTensorType(tensorType, options);
  }
};

} // namespace

void mlir::spn::low::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LoSPNDialect *dialect) {
    SPNBatchExtract::attachInterface<BatchExtractInterface>(*ctx);
    SPNBatchCollect::attachInterface<BatchCollectInterface>(*ctx);
    SPNReturn::attachInterface<ReturnOpInterface>(*ctx);
    SPNTask::attachInterface<TaskInterface>(*ctx);
    SPNKernel::attachInterface<KernelInterface>(*ctx);
    SPNGather::attachInterface<GatherInterface>(*ctx);
  });
}
