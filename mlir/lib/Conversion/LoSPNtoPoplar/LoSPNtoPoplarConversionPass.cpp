//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "../LoSPNtoCPU/NodePatterns.h"
#include "../LoSPNtoCPU/StructurePatterns.h"
#include "LoSPN/LoSPNAttributes.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPNtoCPU/LoSPNtoCPUTypeConverter.h"
#include "LoSPNtoPoplar/LoSPNtoPoplarConversionPasses.h"
#include "mlir-ipu/Dialect/Poplar/IR/Poplar.h"
#include "mlir-ipu/Target/Poplar/TypeTranslation.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <set>
#include <string>

using namespace mlir;
using namespace mlir::spn::low;
using namespace mlir::ipu::poplar;

namespace {

static PoplarTensorType
getPoplarTensorType(Type type, unsigned batchSize,
                    std::optional<unsigned> featureSize = std::nullopt) {
  auto memRefType = type.cast<MemRefType>();
  bool hasDynamicDim = memRefType.isDynamicDim(0) || memRefType.isDynamicDim(1);
  assert(hasDynamicDim && "Batch dimension must be dynamic");
  auto elementType = memRefType.getElementType();
  llvm::SmallVector<int64_t, 4> shape(memRefType.getShape());

  size_t batchDim = memRefType.isDynamicDim(0) ? 0 : 1;
  size_t featureDim = memRefType.isDynamicDim(0) ? 1 : 0;
  shape[batchDim] = batchSize;
  if (featureSize.has_value()) {
    shape[featureDim] = featureSize.value();
  }

  return PoplarTensorType::get(shape, elementType);
}

static CodeletOp translateTask(IRRewriter &rewriter, SPNTask taskOp) {
  IRRewriter::InsertionGuard guard(rewriter);
  Location loc = taskOp->getLoc();
  // Find a unique name for the task
  static int taskCounter = 0;
  std::string taskName = "task" + std::to_string(taskCounter++);

  // Construct the function type
  FunctionType funcType = rewriter.getFunctionType(taskOp->getOperandTypes(),
                                                   taskOp.getResultTypes());
  auto moduleOp = taskOp->getParentOfType<ModuleOp>();
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto codelet = rewriter.create<CodeletOp>(loc, taskName, funcType);

  // Build a loop over the batch index
  rewriter.setInsertionPointToStart(&codelet.getBodyRegion().front());
  auto lb = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  auto inputMemRef = codelet.getBodyRegion().getArgument(0);
  auto inputMemRefTy = inputMemRef.getType().dyn_cast<MemRefType>();
  assert(inputMemRefTy);
  assert(inputMemRefTy.hasRank() && inputMemRefTy.getRank() == 2);
  assert(inputMemRefTy.isDynamicDim(0) ^ inputMemRefTy.isDynamicDim(1));
  auto index = (inputMemRefTy.isDynamicDim(0)) ? 0 : 1;
  auto ub = rewriter.create<memref::DimOp>(loc, inputMemRef, index);
  auto step = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

  auto loop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
  // Collect the values replacing the block values of old block inside the task.
  // The first argument is the batch index, i.e., the loop induction var.
  // The other arguments are the arguments of the entry block of this function.
  SmallVector<Value, 5> blockReplacementArgs;
  blockReplacementArgs.push_back(loop.getInductionVar());
  for (auto bArg : codelet.getBodyRegion().getArguments()) {
    blockReplacementArgs.push_back(bArg);
  }
  // Remove the terminator from the task, as the loop will be terminated by a
  // SCFYieldOp
  rewriter.eraseOp(taskOp.getBody()->getTerminator());

  // Inline the content of the Task into the loop body.
  rewriter.inlineBlockBefore(taskOp.getBody(), loop.getBody()->getTerminator(),
                             blockReplacementArgs);

  // Add a terminator to the codelet
  rewriter.create<::mlir::ipu::poplar::ReturnOp>(loc);

  return codelet;
}

static Value createViewForGatherOp(IRRewriter &rewriter, SPNGather gatherOp,
                                   Value inputVariable, unsigned batchSize) {
  auto inputMemRef = gatherOp.getInput();
  assert(inputMemRef.isa<BlockArgument>() &&
         "Input of gather operation must be a block argument");
  auto inputMemRefType = inputMemRef.getType().cast<BaseMemRefType>();
  auto sliceType =
      getPoplarTensorType(inputMemRefType, batchSize, 1 /*feature size*/);

  SmallVector<int> starts, ends;
  starts.reserve(gatherOp.getIndices().size());
  ends.reserve(gatherOp.getIndices().size());
  SmallVector<Type> resultTypes;
  for (auto index : gatherOp.getIndices()) {
    starts.push_back(index);
    ends.push_back(index + 1);
    resultTypes.push_back(sliceType);
  }
  // The dynamic dimension is the batch dimension, the static dimension is the
  // feature dimension which we want to slice in
  unsigned sliceDim = inputMemRefType.isDynamicDim(0) ? 1 : 0;

  auto sliceOp = rewriter.create<TensorSliceOp>(
      gatherOp.getLoc(), resultTypes, inputVariable, starts, ends, sliceDim);

  auto viewType =
      getPoplarTensorType(gatherOp.getResult().getType(), batchSize);
  auto concatOp = rewriter.create<TensorConcatOp>(
      gatherOp.getLoc(), viewType, sliceOp.getResults(), sliceDim);
  return concatOp.getResult();
}

static FailureOr<Value> translateKernel(IRRewriter &rewriter,
                                        SPNKernel kernelOp, GraphOp graphOp) {
  if (!kernelOp->hasAttrOfType<BSPScheduleAttr>(
          LoSPNDialect::getBSPScheduleAttrName())) {
    kernelOp.emitWarning("Kernel has no schedule attached, skipping");
    return failure();
  }
  assert(kernelOp.getBody().getNumArguments() == 2 &&
         "Kernel currently only supports 2 arguments (input and output)");

  MLIRContext *context = rewriter.getContext();
  auto scheduleAttr = kernelOp->getAttrOfType<BSPScheduleAttr>(
      LoSPNDialect::getBSPScheduleAttrName());
  ProgramType programType = ProgramType::get(context);

  // Translate the tasks and add in-facing and out-facing variables
  DenseMap<unsigned, CodeletOp> codelets; // TaskId -> Codelet
  DenseMap<unsigned, SPNTask> tasks;      // TaskId -> Task

  // MemRef -> out-facing Poplar variable mapping
  // Each memref is mapped to a single out-facing variable because it can only
  // be written to (=defined) by a single task
  DenseMap<Value, Value> outFacingVariableMapping;

  // MemRef -> in-facing Poplar variables mapping
  // Each memref is mapped to a set of in-facing variables because it can be
  // read by multiple tasks
  DenseMap<Value, SmallPtrSet<Value, 4>> inFacingVariableMapping;

  // Sequence that (will) synchronize variables -> MemRefs to be "synchronized"
  // One sequence per superstep
  DenseMap<SequenceOp, std::vector<Value>> copySequences;

  // Create the codelets for the tasks
  for (SPNTask taskOp : kernelOp.getOps<SPNTask>()) {
    assert(taskOp.getTaskId().has_value() &&
           "Task does not have a task id attribute");
    unsigned taskID = taskOp.getTaskId().value();
    auto codelet = translateTask(rewriter, taskOp);
    codelets[taskID] = codelet;
    tasks[taskID] = taskOp;
  }

  // Create the overall sequence
  auto sequenceOp = rewriter.create<SequenceOp>(kernelOp.getLoc(), programType);
  Block &sequenceBlock = sequenceOp.getBodyRegion().emplaceBlock();

  // Schedule the copy from the host-to-device FIFO to a variable representing
  // the input memref
  {
    BlockArgument inputMemRef = kernelOp.getBody().getArgument(0);
    // Create a variable for the input memref
    BaseMemRefType inputMemRefType =
        inputMemRef.getType().cast<BaseMemRefType>();
    PoplarTensorType inputVariableType =
        getPoplarTensorType(inputMemRefType, kernelOp.getBatchSize());
    auto inputVariable = rewriter.create<VariableOp>(
        kernelOp.getLoc(), inputVariableType, "inputTensor", 0);

    // The input variable acts like a out-facing variable, as it defines a value
    // that can be read by multiple tasks
    outFacingVariableMapping[inputMemRef] = inputVariable;

    // Create the input FIFO
    DataStreamType inputFIFOType = DataStreamType::get(inputVariableType);
    auto inputFIFO = rewriter.create<HostToDeviceFIFOOp>(
        kernelOp.getLoc(), inputFIFOType, "inputFIFO");

    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&sequenceBlock);

    rewriter.create<CopyOp>(kernelOp.getLoc(), programType, inputFIFO,
                            inputVariable);

    // BlockArgument outputMemRef = kernelOp.getBody().getArgument(1);
  }

  // Create the supersteps
  std::vector<ComputeSetOp> computeSets;
  computeSets.reserve(scheduleAttr.getSupersteps().size());
  for (const Attribute &attr : scheduleAttr.getSupersteps()) {
    auto superstepAttr = attr.cast<BSPSuperstepAttr>();

    // Create a compute set for the superstep
    auto computeSet = rewriter.create<ComputeSetOp>(
        kernelOp.getLoc(), ComputeSetType::get(rewriter.getContext()));
    Block &computeSetBlock = computeSet.getBodyRegion().emplaceBlock();

    // Memrefs to synchronize after the superstep. Synchronizing means copying
    // the values of associated out-facing to in-facing variables
    std::vector<Value> memrefsToSync;
    memrefsToSync.reserve(superstepAttr.getTaskIds().size());

    // Create the vertices for the superstep
    for (auto [taskIdAP, procIdAP] : llvm::zip(
             superstepAttr.getTaskIds(), superstepAttr.getProcessorIds())) {
      unsigned taskId = taskIdAP.getZExtValue();
      unsigned procId = procIdAP.getZExtValue();
      CodeletOp &codeletOp = codelets[taskId];
      SPNTask taskOp = tasks[taskId];
      SmallVector<Value, 4> variables;

      // Create the in-facing variables for the task
      for (OpOperand &operand :
           taskOp->getOpOperands().drop_back(taskOp.getNumOutArgs())) {
        Value memRef = operand.get();
        if (auto gatherOp = dyn_cast<SPNGather>(memRef.getDefiningOp())) {
          // If this is a (reordered) kernel input, create a view to the kernel
          // input variable instead of a new variable
          Value inputVariable = outFacingVariableMapping[gatherOp.getInput()];
          Value view = createViewForGatherOp(rewriter, gatherOp, inputVariable,
                                             kernelOp.getBatchSize());
          variables.push_back(view);
          continue;
        }

        // Create a new in-facing variable
        std::string name = "task" + std::to_string(taskId) + "_inFacing" +
                           std::to_string(operand.getOperandNumber());

        auto variableType =
            getPoplarTensorType(memRef.getType(), taskOp.getBatchSize());
        auto variable = rewriter.create<VariableOp>(memRef.getLoc(),
                                                    variableType, name, procId);

        variables.push_back(variable);
        inFacingVariableMapping[memRef].insert(variable);
      }

      // Create the out-facing variables
      for (OpOperand &operand :
           taskOp->getOpOperands().take_back(taskOp.getNumOutArgs())) {
        Value memRef = operand.get();
        if (outFacingVariableMapping.count(memRef)) {
          taskOp.emitError("Operand " +
                           std::to_string(operand.getOperandNumber()) +
                           " is already written to by another task");
          return failure();
        }
        std::string name = "task" + std::to_string(taskId) + "_outFacing" +
                           std::to_string(operand.getOperandNumber());

        auto variableType =
            getPoplarTensorType(memRef.getType(), taskOp.getBatchSize());
        auto variable = rewriter.create<VariableOp>(memRef.getLoc(),
                                                    variableType, name, procId);
        variables.push_back(variable);
        outFacingVariableMapping[memRef] = variable;
        memrefsToSync.push_back(memRef);
      }

      // The insertion point is expected to stay at the end of the graph
      IRRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&computeSetBlock);
      rewriter.create<VertexOp>(kernelOp.getLoc(),
                                FlatSymbolRefAttr::get(codeletOp), variables,
                                rewriter.getI16IntegerAttr((int16_t)procId));
    }

    // Schedule the compute set in the sequence
    // The insertion point is expected to stay at the end of the graph
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&sequenceBlock);
    rewriter.create<RunOp>(kernelOp.getLoc(), programType, computeSet);

    // We cannot construct the CopyOp for synchronizing the variables yet as not
    // all in-facing variables correponding to the out-facing variables defined
    // in this superstep may be known at this point.
    // Instead, we add a sequence op for the copies to be scheduled later
    auto copySequence =
        rewriter.create<SequenceOp>(kernelOp.getLoc(), programType);
    copySequences[copySequence] = std::move(memrefsToSync);
  }

  // Create a FIFO for the device-to-host transfer
  {
    BlockArgument outputMemRef = kernelOp.getBody().getArgument(1);
    BaseMemRefType outputMemRefType =
        outputMemRef.getType().cast<BaseMemRefType>();
    PoplarTensorType outputVariableType =
        getPoplarTensorType(outputMemRefType, kernelOp.getBatchSize());
    DataStreamType outputFIFOType = DataStreamType::get(outputVariableType);
    auto outputFIFO = rewriter.create<DeviceToHostFIFOOp>(
        kernelOp.getLoc(), outputFIFOType, "outputFIFO");

    // The output FIFO acts like an in-facing variable
    inFacingVariableMapping[outputMemRef].insert(outputFIFO);
  }

  // Create the CopyOps for synchronizing the variables
  for (auto &[sequenceOp, memrefsToSync] : copySequences) {
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&sequenceOp.getBodyRegion().emplaceBlock());
    for (Value memRef : memrefsToSync) {
      auto inFacingVariables = inFacingVariableMapping[memRef];
      Value outFacingVariable = outFacingVariableMapping[memRef];
      for (Value inFacingVariable : inFacingVariables) {
        rewriter.create<CopyOp>(kernelOp.getLoc(), programType,
                                outFacingVariable, inFacingVariable);
      }
    }
  }

  return sequenceOp.getResult();
}
} // namespace

namespace mlir {
namespace spn {
namespace low {
#define GEN_PASS_DEF_LOSPNTOPOPLARCONVERSIONPASS
#include "LoSPNtoPoplar/LoSPNtoPoplarConversionPasses.h.inc"

struct LoSPNtoPoplarConversionPass
    : public impl::LoSPNtoPoplarConversionPassBase<
          LoSPNtoPoplarConversionPass> {
  using Base::Base;

  FailureOr<GraphOp> translateKernelsToGraph(ModuleOp moduleOp) {
    IRRewriter rewriter(moduleOp->getContext());

    // Create the graph operation
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto graphOp = rewriter.create<ipu::poplar::GraphOp>(moduleOp.getLoc());
    graphOp.getBodyRegion().emplaceBlock();

    rewriter.setInsertionPointToStart(graphOp.getBody());

    llvm::SmallVector<Value> kernelPrograms;
    llvm::SmallVector<SPNKernel> kernelsToErase;
    for (SPNKernel kernelOp : moduleOp.getOps<SPNKernel>()) {
      FailureOr<Value> maybeProgram =
          translateKernel(rewriter, kernelOp, graphOp);
      // Skip the kernel if it failed to translate
      if (failed(maybeProgram))
        continue;

      kernelPrograms.push_back(*maybeProgram);
      kernelsToErase.push_back(kernelOp);
    }

    if (kernelPrograms.empty()) {
      return failure();
    }

    // Add the yield operation to the end of the graph
    rewriter.create<ipu::poplar::YieldOp>(moduleOp.getLoc(),
                                          ValueRange{kernelPrograms});

    // Erase the kernel operations
    for (SPNKernel kernelOp : kernelsToErase) {
      rewriter.eraseOp(kernelOp);
    }
    return graphOp;
  }

  LogicalResult convertNodes() {
    ConversionTarget target(getContext());

    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();
    target.addLegalDialect<mlir::vector::VectorDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::ipu::poplar::PoplarDialect>();
    target.addLegalOp<ModuleOp>();

    LoSPNtoCPUTypeConverter typeConverter;

    target.addIllegalDialect<mlir::spn::low::LoSPNDialect>();

    RewritePatternSet patterns(&getContext());
    mlir::spn::populateLoSPNtoCPUNodePatterns(patterns, &getContext(),
                                              typeConverter);

    auto op = getOperation();
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    return applyFullConversion(op, target, frozenPatterns);
  }

  LogicalResult convertStructure() {
    MLIRContext &context = getContext();
    ConversionTarget target(context);

    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<spn::low::LoSPNDialect>();

    LoSPNtoCPUTypeConverter typeConverter;
    target.addIllegalOp<spn::low::SPNBody>();

    RewritePatternSet patterns(&getContext());

    patterns.insert<BodyLowering>(typeConverter, &context);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    return applyPartialConversion(getOperation(), target, frozenPatterns);
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    FailureOr<GraphOp> maybeGraphOp = translateKernelsToGraph(moduleOp);
    if (failed(maybeGraphOp)) {
      return signalPassFailure();
    }

    if (failed(moduleOp.verify())) {
      return signalPassFailure();
    }

    if (failed(convertStructure())) {
      return signalPassFailure();
    }

    if (failed(convertNodes())) {
      return signalPassFailure();
    }
  }
};
} // namespace low
} // namespace spn
} // namespace mlir