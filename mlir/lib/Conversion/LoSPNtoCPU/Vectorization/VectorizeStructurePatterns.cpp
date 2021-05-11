//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/VectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/SLP/Seeding.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPGraphBuilder.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "../Target/TargetInformation.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

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

  // Apply SLP vectorization.
  //task->getParentOfType<ModuleOp>()->dump();
  task->emitRemark() << "Beginning SLP vectorization";
  auto computationType = operands.back().getType().dyn_cast<MemRefType>().getElementType();
  auto hwVectorWidth = TargetInformation::nativeCPUTarget().getHWVectorEntries(computationType);
  SeedAnalysis seedAnalysis{taskFunc, hwVectorWidth};
  SmallVector<Value, 4> seed;
  seedAnalysis.fillSeed(seed, SearchMode::UseBeforeDef);
  assert(!seed.empty() && "couldn't find a seed!");
  //low::slp::dumpOpTree(seed);
  SLPGraphBuilder builder{3};
  task->emitRemark("Computing graph...");
  auto graph = builder.build(seed);
  auto numNodes = slp::numNodes(graph.get());
  auto numVectors = slp::numVectors(graph.get());
  task->emitRemark("Number of SLP nodes in graph: " + std::to_string(numNodes));
  task->emitRemark("Number of SLP vectors in graph: " + std::to_string(numVectors));
  //low::slp::dumpSLPGraph(graph.get());

  task->emitRemark("Converting graph...");
  // The current vector being transformed.
  NodeVector* vector = nullptr;
  ConversionManager conversionManager{graph.get()};

  // Prevent extracting/removing values more than once (happens in splat mode, if they appear in multiple vectors, ...).
  SmallPtrSet<Value, 32> finishedValues;
  // Use a custom pattern driver that does *not* perform folding automatically (which the other rewrite drivers
  // such as applyOpPatternsAndFold() would do). Folding would mess up identified SLP-vectorizable constants, which
  // aren't necessarily located at the front of the function's body yet. The drivers delete those that aren't at the
  // front (including those we identified as SLP-vectorizable!) and create new ones there, resulting in our constant
  // SLP-vectorization patterns not being applied to them and messing up operand handling further down the SLP graph.
  OwningRewritePatternList patterns;
  populateSLPVectorizationPatterns(patterns, rewriter.getContext(), vector, conversionManager);
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  PatternApplicator applicator(frozenPatterns);
  applicator.applyDefaultCostModel();

  // Track progress.
  auto fivePercent = std::ceil(static_cast<double>(numVectors) * 0.05);
  unsigned fivePercentCounter = 0;

  // Traverse the SLP graph and apply the vectorization patterns.
  auto const& order = conversionManager.conversionOrder();
  for (size_t i = 0; i < order.size(); ++i) {
    vector = order[i];
    //dumpSLPNodeVector(*vector);
    auto* vectorOp = vector->begin()->getDefiningOp();
    if (vector->containsBlockArgs() || failed(applicator.matchAndRewrite(vectorOp, rewriter))) {
      auto const& vectorType = VectorType::get(static_cast<unsigned>(vector->numLanes()), computationType);
      conversionManager.setInsertionPointFor(vector, rewriter);
      // Create extractions from vectorized operands if present.
      for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
        auto const& element = vector->getElement(lane);
        if (auto* elementOp = element.getDefiningOp()) {
          for (size_t i = 0; i < vector->numOperands(); ++i) {
            auto* operand = vector->getOperand(i);
            auto const& source = conversionManager.getValue(operand);
            auto extractOp = rewriter.create<vector::ExtractElementOp>(element.getLoc(), source, lane);
            elementOp->setOperand(i, extractOp.result());
          }
        }
        if (lane == 0 && vector->splattable()) {
          break;
        }
      }
      if (vector->splattable()) {
        auto const& element = vector->getElement(0);
        auto vectorizedOp = rewriter.create<vector::BroadcastOp>(element.getLoc(), vectorType, element);
        conversionManager.update(vector, vectorizedOp, ElementFlag::KeepFirst);
      } else {
        auto vectorizedOp = broadcastFirstInsertRest(vector->begin(), vector->end(), vectorType, rewriter);
        conversionManager.update(vector, vectorizedOp, ElementFlag::KeepAll);
      }
    }
    // Create vector extractions for escaping uses & erase superfluous operations.
    auto const& creationMode = conversionManager.getElementFlag(vector);
    for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
      auto const& element = vector->getElement(lane);
      if (finishedValues.contains(element)) {
        continue;
      }
      finishedValues.insert(element);
      if (creationMode == ElementFlag::KeepAll || (creationMode == ElementFlag::KeepFirst && lane == 0)) {
        continue;
      }
      if (conversionManager.hasEscapingUsers(element)) {
        if (creationMode == ElementFlag::KeepNoneNoExtract) {
          continue;
        }
        rewriter.setInsertionPoint(conversionManager.getEarliestEscapingUser(element));
        auto const& source = conversionManager.getValue(vector);
        auto extractOp = rewriter.create<vector::ExtractElementOp>(element.getLoc(), source, lane);
        if (!source.getDefiningOp()->isBeforeInBlock(extractOp)) {
          llvm::dbgs() << "source: " << source << "\nextract: " << extractOp << "\n";
          assert(false && "extract before source");
        }
        element.replaceAllUsesWith(extractOp.result());
      }
      rewriter.eraseOp(element.getDefiningOp());
    }
    if (static_cast<double>(i) >= (fivePercentCounter * fivePercent)) {
      task->emitRemark("Conversion progress: " + std::to_string(5 * fivePercentCounter++) + '%');
    }
  }
  task->emitRemark("Graph conversion complete.");
  rewriter.restoreInsertionPoint(callPoint);
  rewriter.replaceOpWithNewOp<CallOp>(task, taskFunc, operands);
  taskFunc.body().walk([&](Operation* op) {
    for (auto const& operand : op->getOperands()) {
      if (auto* operandOp = operand.getDefiningOp()) {
        if (op->isBeforeInBlock(operandOp)) {
          op->emitError("DOMINATION PROBLEM: ") << *op << "(operand: " << *operandOp << ")";
        }
      }
    }
    return WalkResult::advance();
  });
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
  auto numSamples = rewriter.create<DimOp>(op.getLoc(), taskBlock->getArgument(0), 0);
  auto vectorWidthConst = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(hwVectorWidth));
  auto remainder = rewriter.create<UnsignedRemIOp>(op.getLoc(), numSamples, vectorWidthConst);
  auto ubVectorized = rewriter.create<SubIOp>(op.getLoc(), numSamples, remainder);

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
