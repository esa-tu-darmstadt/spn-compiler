//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/VectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/TargetInformation.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPSeeding.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPGraphBuilder.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/Support/FormatVariadic.h"

mlir::LogicalResult mlir::spn::VectorizeTask::createFunctionIfVectorizable(low::SPNTask& task,
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
  rewriter.setInsertionPointToStart(task->getParentOfType<mlir::ModuleOp>().getBody());
  SmallVector<Type, 5> inputTypes;
  for (auto operand : operands) {
    inputTypes.push_back(operand.getType());
  }
  auto funcType = FunctionType::get(rewriter.getContext(), inputTypes, {});
  *function = rewriter.create<FuncOp>(task->getLoc(), Twine("vec_task_", std::to_string(taskCount++)).str(), funcType);
  rewriter.restoreInsertionPoint(insertionPoint);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeSingleTask::matchAndRewrite(mlir::spn::low::SPNTask task,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {

  if (task.batchSize() > 1) {
    return rewriter.notifyMatchFailure(task,
                                       "Specialized for single batch vectorization, does not match for batchSize > 1");
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
  for (auto bArg : taskBlock->getArguments()) {
    blockReplacementArgs.push_back(bArg);
  }
  // Inline the content of the Task into the function.
  rewriter.mergeBlocks(&task.body().front(), taskBlock, blockReplacementArgs);

  // Apply SLP vectorization.
  task->emitRemark() << "Computing new seed for SLP vectorization...";
  auto computationType = operands.back().getType().dyn_cast<MemRefType>().getElementType();
  auto hwVectorWidth = TargetInformation::nativeCPUTarget().getHWVectorEntries(computationType);
  low::slp::SeedAnalysis seedAnalysis{taskFunc};
  auto const& seed = seedAnalysis.getSeed(hwVectorWidth, low::slp::SearchMode::UseBeforeDef);
  assert(!seed.empty() && "couldn't find a seed!");
  // low::slp::dumpOpTree(seed);
  // seed.front().getDefiningOp()->getBlock()->dump();
  low::slp::SLPGraphBuilder builder{3};
  auto graph = builder.build(seed);
  // low::slp::dumpSLPGraph(*graph);

  task->emitRemark() << "Transforming SLP graph back into vectorized ops...";
  // Maps individual SLP node vectors to finished vector operations.
  DenseMap<low::slp::NodeVector*, Value> vectorizedOps;
  // Marks operations as potentially erasable (potentially because they might have escaping uses).
  SmallPtrSet<low::slp::NodeVector*, 32> erasableOps;
  // Stores escaping uses for vector extractions that might be necessary later on.
  auto const& escapingLanes = low::slp::SLPNode::escapingLanesMap(graph.get());
  // Prevent extracting/removing values more than once (happens in splat mode, if they appear in multiple vectors, ...).
  SmallPtrSet<Value, 32> finishedValues;
  // The current vector being transformed.
  low::slp::NodeVector* vector = nullptr;

  OwningRewritePatternList patterns;
  populateSLPVectorizationPatterns(patterns, rewriter.getContext(), vector, vectorizedOps, erasableOps);
  FrozenRewritePatternList frozenPatterns(std::move(patterns));

  // Use a custom pattern driver that does *not* perform folding automatically (which the other rewrite drivers
  // such as applyOpPatternsAndFold() would do). Folding would mess up identified SLP-vectorizable constants, which
  // aren't necessarily located at the front of the function's body yet. The drivers delete those that aren't at the
  // front (including those we identified as SLP-vectorizable!) and create new ones there, resulting in our constant
  // SLP-vectorization patterns not being applied to them and messing up operand handling further down the SLP graph.
  PatternApplicator applicator(frozenPatterns);
  applicator.applyDefaultCostModel();

  // Traverse the SLP graph in postorder and apply the vectorization patterns.
  for (auto* node : low::slp::SLPNode::postOrder(graph.get())) {

    // Also traverse nodes in postorder to properly handle multinodes.
    for (size_t vectorIndex = node->numVectors(); vectorIndex-- > 0;) {
      vector = node->getVector(vectorIndex);
      // This assumes that there exist other vectors that use the current node vector directly (or don't need it), e.g.
      // batch reads that get lowered to vector loads instead of using a vector of indices as input.
      if (!vector->vectorizable()) {
        continue;
      }
      if (vector->containsBlockArgs()) {
        auto const& type = VectorType::get(static_cast<unsigned>(vector->numLanes()), computationType);
        auto const& insertionPoint = rewriter.saveInsertionPoint();
        rewriter.setInsertionPoint(low::slp::firstUser(vector->begin(), vector->end()));
        auto vectorOp = low::slp::broadcastFirstInsertRest(vector->begin(), vector->end(), type, rewriter);
        rewriter.restoreInsertionPoint(insertionPoint);
        vectorizedOps[vector] = vectorOp;
      } else {
        auto* vectorOp = vector->begin()->getDefiningOp();
        if (failed(applicator.matchAndRewrite(vectorOp, rewriter))) {
          vectorOp->emitOpError("SLP pattern application failed (did you forget to specify the pattern?)");
        }
      }
    }

    // Create vector extractions for escaping uses & erase superfluous operations.
    for (size_t vectorIndex = node->numVectors(); vectorIndex-- > 0;) {
      vector = node->getVector(vectorIndex);
      if (vectorizedOps.count(vector) && erasableOps.count(vector)) {
        for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
          auto const& vectorValue = vector->getElement(lane);
          if (finishedValues.contains(vectorValue)) {
            continue;
          }
          if (escapingLanes.count(vector) && escapingLanes.lookup(vector)->contains(lane)) {
            auto const& source = vectorizedOps[vector];
            rewriter.setInsertionPoint(low::slp::firstOccurrence(std::begin(vectorValue.getUsers()),
                                                                 std::end(vectorValue.getUsers())));
            auto extractOp = rewriter.create<mlir::vector::ExtractElementOp>(vectorValue.getLoc(), source, lane);
            vectorValue.replaceAllUsesWith(extractOp.result());
          } else {
            vectorValue.replaceAllUsesWith(vectorizedOps[vector]);
          }
          finishedValues.insert(vectorValue);
          rewriter.eraseOp(vectorValue.getDefiningOp());
        }
      }
    }
  }

  rewriter.restoreInsertionPoint(callPoint);
  rewriter.replaceOpWithNewOp<CallOp>(task, taskFunc, operands);
  return success();

}

mlir::LogicalResult mlir::spn::VectorizeBatchTask::matchAndRewrite(mlir::spn::low::SPNTask op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  static int taskCount = 0;

  if (op.batchSize() <= 1) {
    return rewriter.notifyMatchFailure(op,
                                       "Specialized for batch vectorization, does not match for batchSize == 1");
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
    rewriter.clone(node, mapVectorTaskArgs);
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
