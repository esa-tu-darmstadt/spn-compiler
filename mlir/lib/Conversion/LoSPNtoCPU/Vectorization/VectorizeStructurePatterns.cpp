//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/VectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/SLP/Analysis.h"
#include "LoSPNtoCPU/Vectorization/SLP/Seeding.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPPatternMatch.h"
#include "../Target/TargetInformation.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

namespace {
  SmallPtrSet<Operation*, 32> computeDeadOps(Block* block) {
    SmallPtrSet<Operation*, 32> deadOps;
    llvm::SmallSetVector<Operation*, 32> worklist;
    block->walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (isOpTriviallyDead(op)) {
        worklist.insert(op);
        deadOps.insert(op);
      }
    });
    while (!worklist.empty()) {
      auto* op = worklist.pop_back_val();
      for (auto const& operand : op->getOperands()) {
        if (auto* operandOp = operand.getDefiningOp()) {
          auto users = operandOp->getUsers();
          if (std::all_of(std::begin(users), std::end(users), [&](Operation* user) {
            return deadOps.contains(user);
          })) {
            worklist.insert(operandOp);
            deadOps.insert(operandOp);
          }
        }
      }
    }
    return deadOps;
  }
}

LogicalResult VectorizeSingleTask::matchAndRewrite(SPNTask task,
                                                   llvm::ArrayRef<Value> operands,
                                                   ConversionPatternRewriter& rewriter) const {

  if (task.batchSize() > 1) {
    return rewriter.notifyMatchFailure(task, "SLP vectorization does not match for batchSize > 1");
  }

  if (task.body().getBlocks().size() > 1) {
    return rewriter.notifyMatchFailure(task, "SLP vectorization only applicable to single basic blocks (yet)");
  }

  auto const& callPoint = rewriter.saveInsertionPoint();

  FuncOp taskFunc;
  if (failed(createFunctionIfVectorizable(task, operands, rewriter, &taskFunc))) {
    return failure();
  }

  auto taskBlock = taskFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(taskBlock);

  // Collect the values replacing the block values of old block inside the task.
  // The first argument is the batch index, in this case (for a single execution),
  // we can simply set it to constant zero.
  // The other arguments are the arguments of the entry block of this function.
  SmallVector<Value, 5> blockReplacementArgs;
  blockReplacementArgs.push_back(rewriter.create<ConstantOp>(task.getLoc(), rewriter.getIndexAttr(0)));
  for (auto const& bArg : taskBlock->getArguments()) {
    blockReplacementArgs.push_back(bArg);
  }
  // Inline the content of the task into the function.
  rewriter.mergeBlocks(&task.body().front(), taskBlock, blockReplacementArgs);
  // Replace block arguments *now* because moving operations later on somehow 'resets' their block argument operands
  // and does not remap them in the end, which leads to failures if a block argument should be erased.
  taskBlock->walk([&](Operation* op) {
    for (size_t i = 0; i < op->getNumOperands(); ++i) {
      op->setOperand(i, rewriter.getRemappedValue(op->getOperand(i)));
    }
  });

  // Apply SLP vectorization.
  task->emitRemark() << "Beginning SLP vectorization...";
  task->emitRemark() << "Total number of operations in function: " << taskFunc.body().front().getOperations().size();
  // We don't want to carry thousands of needlessly created operations with us to the next vectorization attempt.
  // Therefore, use an IRRewriter that erases operations *immediately* instead of at the end of the conversion process
  // (as would be the case for PatterRewriter::eraseOp()).
  IRRewriter graphRewriter{rewriter};
  auto costModel = std::make_shared<UnitCostModel>();
  ConversionManager conversionManager{graphRewriter, taskBlock, costModel};

  SmallVector<std::unique_ptr<SLPVectorizationPattern>, 10> patterns;
  populateSLPVectorizationPatterns(patterns, conversionManager);
  SLPPatternApplicator applicator{costModel, std::move(patterns)};

  auto elementType = operands.back().getType().dyn_cast<MemRefType>().getElementType();
  std::unique_ptr<SeedAnalysis> seedAnalysis;
  {
    bool topDown = true;
    auto width = TargetInformation::nativeCPUTarget().getHWVectorEntries(elementType);
    if (topDown) {
      seedAnalysis = std::make_unique<TopDownAnalysis>(taskFunc, width);
    } else {
      seedAnalysis = std::make_unique<FirstRootAnalysis>(taskFunc, width);
    }
  }

  auto currentFunctionCost = costModel->getBlockCost(taskBlock, computeDeadOps(taskBlock));
  for (auto seed = seedAnalysis->next(); !seed.empty(); seed = seedAnalysis->next()) {
    task->emitRemark("Computing graph...");
    SLPGraph graph{seed, 3};

    // ============================================================================================================= //
    {
      bool dependencyStuff = false;
      if (dependencyStuff) {
        auto dependencyGraph = graph.dependencyGraph();
        task->emitRemark("#Nodes in dependency graph: ") << dependencyGraph.numNodes();
        task->emitRemark("#Edges in dependency graph: ") << dependencyGraph.numEdges();
        dumpDependencyGraph(dependencyGraph);
        for (auto* superword : dependencyGraph.postOrder()) {
          dumpSuperword(*superword);
        }
        break;
      }
      bool analysisStuff = false;
      if (analysisStuff) {
        analyzeTopologicalMixing(graph);
        for (auto& op : taskBlock->getOperations()) {
          rewriter.eraseOp(&op);
        }
        graphRewriter.setInsertionPointToStart(taskBlock);
        graphRewriter.create<ReturnOp>(task->getLoc());
        break;
      }
    }
    // ============================================================================================================= //

    task->emitRemark("Converting graph...");
    auto order = conversionManager.startConversion(graph);
    // Traverse the SLP graph and apply the vectorization patterns.
    auto numVectors = order.size();
    task->emitRemark("Number of SLP vectors in graph: " + std::to_string(numVectors));
    task->emitRemark("Number of unique ops in graph:  " + std::to_string(numUniqueOps(order)));
    for (auto* superword : order) {
      applicator.matchAndRewrite(superword, graphRewriter);
    }
    task->emitRemark("Conversion complete.");
    auto vectorizedFunctionCost = costModel->getBlockCost(taskBlock, computeDeadOps(taskBlock));
    if (vectorizedFunctionCost >= currentFunctionCost) {
      task->emitRemark("Vectorization not profitable, reverting back to non-vectorized state.");
      conversionManager.cancelConversion();
    } else {
      task->emitRemark("Vectorization profitable, keeping vectorized state.");
      conversionManager.finishConversion();
      seedAnalysis->update(order);
      currentFunctionCost = vectorizedFunctionCost;
    }
  }
  task->emitRemark("SLP vectorization complete.");
  rewriter.restoreInsertionPoint(callPoint);
  rewriter.replaceOpWithNewOp<CallOp>(task, taskFunc, operands);
  return success();
}

LogicalResult VectorizeBatchTask::matchAndRewrite(SPNTask op,
                                                  llvm::ArrayRef<Value> operands,
                                                  ConversionPatternRewriter& rewriter) const {

  if (op.batchSize() <= 1) {
    return rewriter.notifyMatchFailure(op, "Specialized for batch vectorization, does not match for batchSize == 1");
  }

  auto restore = rewriter.saveInsertionPoint();

  FuncOp taskFunc;
  if (failed(createFunctionIfVectorizable(op, operands, rewriter, &taskFunc))) {
    return failure();
  }

  assert(operands.back().getType().isa<MemRefType>());
  auto computationType = operands.back().getType().dyn_cast<MemRefType>().getElementType();
  auto hwVectorWidth = TargetInformation::nativeCPUTarget().getHWVectorEntries(computationType);

  // Emit a warning if the target vector width does not divide the requested batch size.
  // This will cause a part of each batch (batchSize % vectorWidth elements) to be processed
  // by the scalar epilog loop instead of the vectorized loop.
  if ((op.batchSize() % hwVectorWidth) != 0) {
    op.emitWarning() << "The target vector width " << hwVectorWidth << " does not divide the requested batch size "
                     << op.batchSize() << "; This can result in degraded performance. "
                     << "Choose the batch size as a multiple of the vector width " << hwVectorWidth;
  }

  auto taskBlock = taskFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(taskBlock);
  auto numSamples = rewriter.create<memref::DimOp>(op.getLoc(), taskBlock->getArgument(0), 0);
  auto vectorWidthConst = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(hwVectorWidth));
  auto remainder = rewriter.create<mlir::UnsignedRemIOp>(op.getLoc(), numSamples, vectorWidthConst);
  auto ubVectorized = rewriter.create<mlir::SubIOp>(op.getLoc(), numSamples, remainder);

  // Create the vectorized loop, iterating from 0 to ubVectorized, in steps of hwVectorWidth.
  auto lbVectorized = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  auto stepVectorized = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(hwVectorWidth));
  auto vectorizedLoop = rewriter.create<scf::ForOp>(op.getLoc(), lbVectorized, ubVectorized, stepVectorized);
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
  auto stepScalar = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1));
  auto scalarLoop = rewriter.create<scf::ForOp>(op.getLoc(), ubVectorized, numSamples, stepScalar);
  auto& scalarLoopBody = scalarLoop.getLoopBody().front();

  restoreTask = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&scalarLoopBody);
  SmallVector<Value, 5> blockReplacementArgs;
  blockReplacementArgs.push_back(scalarLoop.getInductionVar());
  for (auto bArg : taskBlock->getArguments()) {
    blockReplacementArgs.push_back(bArg);
  }
  rewriter.mergeBlockBefore(&op.body().front(), scalarLoopBody.getTerminator(), blockReplacementArgs);
  scalarLoopBody.walk([&rewriter](SPNReturn ret) {
    assert(ret.returnValues().empty() && "Task return should be empty");
    rewriter.eraseOp(ret);
  });

  rewriter.restoreInsertionPoint(restoreTask);
  rewriter.create<ReturnOp>(op->getLoc());
  // Insert a call to the newly created task function.
  rewriter.restoreInsertionPoint(restore);
  rewriter.replaceOpWithNewOp<CallOp>(op, taskFunc, operands);
  return success();

}

LogicalResult VectorizeTask::createFunctionIfVectorizable(SPNTask& task,
                                                          ArrayRef<Value> const& operands,
                                                          ConversionPatternRewriter& rewriter,
                                                          FuncOp* function) const {
  static int taskCount = 0;

  assert(operands.back().getType().isa<MemRefType>());
  auto computationType = operands.back().getType().dyn_cast<MemRefType>().getElementType();
  auto hwVectorWidth = TargetInformation::nativeCPUTarget().getHWVectorEntries(computationType);

  if (hwVectorWidth <= 1) {
    return rewriter.notifyMatchFailure(task,
                                       llvm::formatv(
                                           "No vectorization possible for data-type {} on the requested target",
                                           computationType));
  }

  if (requireAllOpsVectorizable) {
    // Check if all nodes can be vectorized before trying to do so.
    auto allVectorizable = task.body().walk([hwVectorWidth](low::LoSPNVectorizable vOp) {
      if (!vOp.isVectorizable(hwVectorWidth)) {
        vOp.emitRemark() << "Operation cannot be vectorized with vector width " << hwVectorWidth;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (allVectorizable.wasInterrupted()) {
      return rewriter.notifyMatchFailure(task, "Not all nested operations can be vectorized, aborting vectorization");
    }
  }

  // Let the user know which vector width will be used.
  task->emitRemark() << "Attempting to vectorize with vector width " << hwVectorWidth << " for data-type "
                     << computationType;

  auto const& insertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(task->getParentOfType<ModuleOp>().getBody());
  SmallVector<Type, 5> inputTypes;
  for (auto operand : operands) {
    inputTypes.push_back(operand.getType());
  }
  auto funcType = FunctionType::get(rewriter.getContext(), inputTypes, {});
  *function = rewriter.create<FuncOp>(task->getLoc(), Twine("vec_task_", std::to_string(taskCount++)).str(), funcType);
  rewriter.restoreInsertionPoint(insertionPoint);
  return success();
}
