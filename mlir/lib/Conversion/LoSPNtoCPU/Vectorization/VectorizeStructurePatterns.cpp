//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <chrono>

#include "LoSPNtoCPU/Vectorization/VectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/SLP/CostModel.h"
#include "LoSPNtoCPU/Vectorization/SLP/Seeding.h"
#include "../Target/TargetInformation.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

// =======================================================================================================//
namespace {
  typedef std::chrono::high_resolution_clock::time_point TimePoint;
}
// =======================================================================================================//
LogicalResult VectorizeSingleTask::matchAndRewrite(SPNTask task,
                                                   VectorizeSingleTask::OpAdaptor adaptor,
                                                   ConversionPatternRewriter& rewriter) const {

  auto operands = adaptor.getOperands();

  if (task.getBatchSize() > 1) {
    return rewriter.notifyMatchFailure(task, "SLP vectorization does not match for batchSize > 1");
  }

  if (task.getBody().getBlocks().size() > 1) {
    return rewriter.notifyMatchFailure(task, "SLP vectorization only applicable to single basic blocks");
  }

  auto const& callPoint = rewriter.saveInsertionPoint();

  func::FuncOp taskFunc;
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
  blockReplacementArgs.push_back(rewriter.create<arith::ConstantOp>(task.getLoc(), rewriter.getIndexAttr(0)));
  for (auto const& bArg : taskBlock->getArguments()) {
    blockReplacementArgs.push_back(bArg);
  }
  // Inline the content of the task into the function.
  rewriter.mergeBlocks(&task.getBody().front(), taskBlock, blockReplacementArgs);
  // Replace block arguments *now* because moving operations later on somehow 'resets' their block argument operands
  // and does not remap them in the end, which leads to failures if a block argument should be erased.
  taskBlock->walk([&](Operation* op) {
    for (size_t i = 0; i < op->getNumOperands(); ++i) {
      op->setOperand(i, rewriter.getRemappedValue(op->getOperand(i)));
    }
  });

  // Apply SLP vectorization.
  task->emitRemark() << "Beginning SLP vectorization (max attempts: " << maxAttempts
                     << ", max successful iterations: " << maxSuccessfulIterations
                     << ", max multinode size: " << maxNodeSize
                     << ", max look-ahead: " << maxLookAhead
                     << ", reordering DFS: " << reorderInstructionsDFS
                     << ", duplicates allowed: " << allowDuplicateElements
                     << ", topological mixing allowed: " << allowTopologicalMixing
                     << ", use XOR chains: " << useXorChains << ").";

#ifndef SLP_DEBUG
  #define SLP_DEBUG false
#endif

// Print the number of loSPN ops in the entire function
#define PRINT_SIZE SLP_DEBUG
// Count how often each opcode appears in the entire function and print it.
#define PRINT_OP_STATS SLP_DEBUG && false
// Print how much of the original function has been covered by all SLP graphs combined
#define PRINT_SLP_COVER SLP_DEBUG
#define PRINT_SLP_GRAPH_SIZE SLP_DEBUG
#define PRINT_SLP_GRAPH_NODE_SIZES SLP_DEBUG && false
#define PRINT_SUCCESSFUL_ITERATION_COUNT SLP_DEBUG

#define DEPENDENCY_ANALYSIS SLP_DEBUG && false
#define COST_MODEL_ANALYSIS SLP_DEBUG && false

// Prints how much time each step took (seeding, graph building, pattern matching, ...)
#define PRINT_TIMINGS SLP_DEBUG
#define TOP_DOWN_SEEDING true

#if PRINT_SIZE
  unsigned liveOps = taskBlock->getOperations().size() - computeDeadOps(taskBlock).size();
  llvm::outs() << "#ops before vectorization: " << std::to_string(liveOps) << "\n";
#endif
#if PRINT_OP_STATS
  llvm::StringMap<unsigned> opCounts;
  taskBlock->walk([&](Operation* op) {
    ++opCounts[op->getName().getStringRef()];
  });
  for (auto const& entry : opCounts) {
    llvm::outs() << "OPCOUNT: " << entry.first() << ", count: " << entry.second << "\n";
  }
#endif
#if PRINT_SLP_COVER
  SmallPtrSet<Operation*, 32> allOps;
  taskBlock->walk([&](Operation* op) {
    allOps.insert(op);
  });
#endif

  // In case an SLP graph is not deemed profitable, we don't want to carry thousands of needlessly created operations
  // with us to the next vectorization attempt.
  // Therefore, use an IRRewriter that erases operations *immediately* instead of at the end of the conversion process
  // (as would be the case for ConversionPatterRewriter::eraseOp()).
  IRRewriter graphRewriter{rewriter};

  CostModelPatternApplicator<UnitCostModel> applicator;
  auto* costModel = applicator.getCostModel();

  ConversionManager conversionManager{graphRewriter, taskBlock, costModel, reorderInstructionsDFS};
  applicator.setPatterns(allSLPVectorizationPatterns(conversionManager));

  auto elementType = operands.back().getType().dyn_cast<MemRefType>().getElementType();
  std::unique_ptr<SeedAnalysis> seedAnalysis;
  auto width = TargetInformation::nativeCPUTarget().getHWVectorEntries(elementType);
#if TOP_DOWN_SEEDING
  seedAnalysis = std::make_unique<TopDownAnalysis>(taskFunc, width);
#else
  seedAnalysis = std::make_unique<FirstRootAnalysis>(taskFunc, width);
#endif

  auto currentFunctionCost = costModel->getBlockCost(taskBlock, computeDeadOps(taskBlock));

#if PRINT_TIMINGS
  TimePoint totalStart = std::chrono::high_resolution_clock::now();
#endif

  SmallVector<Value, 4> seed;
  unsigned successfulIterations = 0;
  unsigned attempts = 0;
  while (successfulIterations < maxSuccessfulIterations && attempts++ < maxAttempts) {

#if PRINT_TIMINGS
    TimePoint seedStart = std::chrono::high_resolution_clock::now();
#endif

    seed.assign(seedAnalysis->next());
    if (seed.empty()) {
      break;
    }

#if PRINT_TIMINGS
    TimePoint seedEnd = std::chrono::high_resolution_clock::now();
    auto seedDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(seedEnd - seedStart);
#endif
#if COST_MODEL_ANALYSIS
    auto line = Twine("SLP Iteration: ").concat(std::to_string(successfulIterations)).str();
    appendLineToFile("costAnalysis.log", line);
    line = Twine("Estimated Cost: ").concat(std::to_string(currentFunctionCost)).str();
    appendLineToFile("costAnalysis.log", line);
#endif
#if PRINT_TIMINGS
    TimePoint graphStart = std::chrono::high_resolution_clock::now();
#endif

    SLPGraph graph{seed, maxNodeSize, maxLookAhead, allowDuplicateElements, allowTopologicalMixing, useXorChains};

#if PRINT_TIMINGS
    TimePoint graphEnd = std::chrono::high_resolution_clock::now();
    auto graphDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(graphEnd - graphStart);
#endif
#if PRINT_SLP_GRAPH_NODE_SIZES
    DenseMap<unsigned, unsigned> nodeSizes;
    graph::walk(graph.getRootNode().get(), [&](SLPNode* node) {
      ++nodeSizes[node->numSuperwords()];
    });
    for (auto const& entry : nodeSizes) {
      llvm::outs() << "NODE SIZE (" << successfulIterations << "): " << entry.first << ", count: " << entry.second
      << "\n";
    }
#endif
#if DEPENDENCY_ANALYSIS
    auto dependencyGraph = graph.dependencyGraph();
    llvm::outs() << "#Nodes in dependency graph: " << dependencyGraph.numNodes() << "\n";
    llvm::outs() << "#Edges in dependency graph: " << dependencyGraph.numEdges() << "\n";
#endif

    auto order = conversionManager.startConversion(graph);

#if PRINT_SLP_GRAPH_SIZE
    unsigned numSuperwords = 0;
    for (auto* superword : order) {
      if (!superword->constant()) {
        ++numSuperwords;
      }
    }
    llvm::outs() << "#superwords in graph (" << successfulIterations << "): " << numSuperwords << "\n";
    SmallPtrSet<Operation*, 32> uniqueOps;
    for (auto* superword: order) {
      if (superword->constant()) {
        continue;
      }
      for (auto value : *superword) {
        if (auto definingOp = value.getDefiningOp()) {
          if (dyn_cast<SPNBatchRead>(definingOp)) {
            continue;
          }
          uniqueOps.insert(definingOp);
        }
      }
    }
    llvm::outs() << "#unique arithmetic graph ops (" << successfulIterations << "): " << uniqueOps.size() << "\n";
#endif

#if PRINT_TIMINGS
    TimePoint rewriteStart = std::chrono::high_resolution_clock::now();
#endif

    // Traverse the SLP graph and apply the vectorization patterns.
    for (auto* superword : order) {
      applicator.matchAndRewrite(superword, graphRewriter);
    }

#if PRINT_TIMINGS
    TimePoint rewriteEnd = std::chrono::high_resolution_clock::now();
    auto rewriteDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(rewriteEnd - rewriteStart);
#endif

    auto vectorizedFunctionCost = costModel->getBlockCost(taskBlock, computeDeadOps(taskBlock));
    // Vectorization not profitable.
    if (vectorizedFunctionCost >= currentFunctionCost) {
      conversionManager.cancelConversion();
    }
      // Vectorization profitable.
    else {
      conversionManager.finishConversion();
      seedAnalysis->update(order);
      currentFunctionCost = vectorizedFunctionCost;

#if PRINT_SLP_COVER
      auto deadOps = computeDeadOps(taskBlock);
      unsigned coveredOps = 0;
      for (auto* op : allOps) {
        if (deadOps.contains(op)) {
          ++coveredOps;
        }
      }
      double percentage = static_cast<double>(coveredOps * 100) / allOps.size();
      llvm::outs() << "% function ops dead (" << successfulIterations << "): " << percentage << "%\n";
#endif
#if PRINT_TIMINGS
      llvm::outs() << "SEED TIME (" << successfulIterations << "): " << seedDuration.count() << " ns\n";
      llvm::outs() << "GRAPH TIME (" << successfulIterations << "): " << graphDuration.count() << " ns\n";
      llvm::outs() << "PATTERN REWRITE TIME (" << successfulIterations << "): " << rewriteDuration.count() << " ns\n";
#endif

      ++successfulIterations;
    }
  }
  task->emitRemark("SLP vectorization complete.");

#if PRINT_TIMINGS
  TimePoint totalEnd = std::chrono::high_resolution_clock::now();
  auto totalDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(totalEnd - totalStart);
  llvm::outs() << "SLP TOTAL TIME: " << totalDuration.count() << " ns\n";
#endif
#if PRINT_SIZE
  liveOps = taskBlock->getOperations().size() - computeDeadOps(taskBlock).size();
  llvm::outs() << "#ops after vectorization (total): " << taskBlock->getOperations().size() << "\n";
  llvm::outs() << "#ops after vectorization (not dead): " << liveOps << "\n";
#endif
#if PRINT_SUCCESSFUL_ITERATION_COUNT
  llvm::outs() << "profitable SLP iterations: " << successfulIterations << "\n";
#endif

  // A lot of operations won't be needed anymore if we vectorized at least once.
  if (successfulIterations > 0) {
    for (auto* op : computeDeadOps(taskBlock)) {
      rewriter.eraseOp(op);
    }
  }
  rewriter.restoreInsertionPoint(callPoint);
  rewriter.replaceOpWithNewOp<func::CallOp>(task, taskFunc, operands);
  return success();
}

LogicalResult VectorizeBatchTask::matchAndRewrite(SPNTask op,
                                                  VectorizeBatchTask::OpAdaptor adaptor,
                                                  ConversionPatternRewriter& rewriter) const {

  auto operands = adaptor.getOperands();
  
  if (op.getBatchSize() <= 1) {
    return rewriter.notifyMatchFailure(op, "Specialized for batch vectorization, does not match for batchSize == 1");
  }

  auto restore = rewriter.saveInsertionPoint();

  func::FuncOp taskFunc;
  if (failed(createFunctionIfVectorizable(op, operands, rewriter, &taskFunc))) {
    return failure();
  }

  assert(operands.back().getType().isa<MemRefType>());
  auto computationType = operands.back().getType().dyn_cast<MemRefType>().getElementType();
  auto hwVectorWidth = TargetInformation::nativeCPUTarget().getHWVectorEntries(computationType);

  // Emit a warning if the target vector width does not divide the requested batch size.
  // This will cause a part of each batch (batchSize % vectorWidth elements) to be processed
  // by the scalar epilog loop instead of the vectorized loop.
  if ((op.getBatchSize() % hwVectorWidth) != 0) {
    op.emitWarning() << "The target vector width " << hwVectorWidth << " does not divide the requested batch size "
                     << op.getBatchSize() << "; This can result in degraded performance. "
                     << "Choose the batch size as a multiple of the vector width " << hwVectorWidth;
  }

  auto taskBlock = taskFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(taskBlock);
  // The number of samples can be derived from the dynamic dimension of one of the input memrefs.
  auto inputMemRef = taskBlock->getArgument(0);
  auto inputMemRefTy = inputMemRef.getType().dyn_cast<MemRefType>();
  assert(inputMemRefTy);
  assert(inputMemRefTy.hasRank() && inputMemRefTy.getRank() == 2);
  assert(inputMemRefTy.isDynamicDim(0) ^ inputMemRefTy.isDynamicDim(1));
  auto index = (inputMemRefTy.isDynamicDim(0)) ? 0 : 1;
  auto numSamples = rewriter.create<memref::DimOp>(op.getLoc(), inputMemRef, index);
  auto vectorWidthConst = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(hwVectorWidth));
  auto remainder = rewriter.create<arith::RemUIOp>(op.getLoc(), numSamples, vectorWidthConst);
  auto ubVectorized = rewriter.create<arith::SubIOp>(op.getLoc(), numSamples, remainder);

  // Create the vectorized loop, iterating from 0 to ubVectorized, in steps of hwVectorWidth.
  auto lbVectorized = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  auto stepVectorized = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(hwVectorWidth));
  auto vectorizedLoop = rewriter.create<scf::ForOp>(op.getLoc(), lbVectorized, ubVectorized, stepVectorized);
  auto& vectorLoopBody = vectorizedLoop.getLoopBody().front();

  auto restoreTask = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&vectorLoopBody);
  auto oldTaskArgs = op.getBody().front().getArguments();
  BlockAndValueMapping mapVectorTaskArgs;
  // Map from batchIndex to vectorized loop induction var.
  mapVectorTaskArgs.map(oldTaskArgs.front(), vectorizedLoop.getInductionVar());
  int i = 1;
  for (auto bArg : taskBlock->getArguments()) {
    mapVectorTaskArgs.map(oldTaskArgs[i++], bArg);
  }
  // Copy the operations from the Task's content to the vectorized loop
  for (auto& node : op.getBody().front()) {
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
  auto stepScalar = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1));
  auto scalarLoop = rewriter.create<scf::ForOp>(op.getLoc(), ubVectorized, numSamples, stepScalar);
  auto& scalarLoopBody = scalarLoop.getLoopBody().front();

  restoreTask = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&scalarLoopBody);
  SmallVector<Value, 5> blockReplacementArgs;
  blockReplacementArgs.push_back(scalarLoop.getInductionVar());
  for (auto bArg : taskBlock->getArguments()) {
    blockReplacementArgs.push_back(bArg);
  }
  rewriter.mergeBlockBefore(&op.getBody().front(), scalarLoopBody.getTerminator(), blockReplacementArgs);
  scalarLoopBody.walk([&rewriter](SPNReturn ret) {
    assert(ret.getReturnValues().empty() && "Task return should be empty");
    rewriter.eraseOp(ret);
  });

  rewriter.restoreInsertionPoint(restoreTask);
  rewriter.create<func::ReturnOp>(op->getLoc());
  // Insert a call to the newly created task function.
  rewriter.restoreInsertionPoint(restore);
  rewriter.replaceOpWithNewOp<func::CallOp>(op, taskFunc, operands);
  return success();

}

LogicalResult VectorizeTask::createFunctionIfVectorizable(SPNTask& task,
                                                          ValueRange operands,
                                                          ConversionPatternRewriter& rewriter,
                                                          func::FuncOp* function) const {
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
    auto allVectorizable = task.getBody().walk([hwVectorWidth](low::LoSPNVectorizable vOp) {
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
  *function = rewriter.create<func::FuncOp>(task->getLoc(), Twine("vec_task_", std::to_string(taskCount++)).str(), funcType);
  rewriter.restoreInsertionPoint(insertionPoint);
  return success();
}
