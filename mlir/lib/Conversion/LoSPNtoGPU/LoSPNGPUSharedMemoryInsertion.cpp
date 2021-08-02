//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "LoSPNtoGPU/LoSPNtoGPUConversionPasses.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "Target/CUDATargetInformation.h"

using namespace mlir;
using namespace mlir::spn;

class RewriteBatchReadtoSharedMem : public mlir::OpRewritePattern<low::SPNBatchRead> {

public:

  RewriteBatchReadtoSharedMem(mlir::MLIRContext* ctx, Value sharedMemory, unsigned minIndex) :
      OpRewritePattern{ctx, 1}, sharedMem{sharedMemory}, minIdx{minIndex} {}

  LogicalResult matchAndRewrite(low::SPNBatchRead op, PatternRewriter& rewriter) const override {
    //
    // Replace a BatchRead, whose input memory has been preloaded to shared memory with an ordinary load.
    // As the input is transposed during pre-loading, the indices need to be flipped, with the
    // featureIndex indexing the row and the batchIndex (threadID.x) the column.
    auto loc = op->getLoc();
    auto threadID = rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(),
                                                     rewriter.getStringAttr("x"));
    auto featureIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(op.staticIndex() - minIdx));
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, sharedMem, ValueRange{featureIndex, threadID});
    return success();
  }

private:

  Value sharedMem;

  unsigned minIdx;

};

int getConstantBlockSize(gpu::LaunchFuncOp launch) {
  //
  // Check whether a LaunchFunc operation uses a constant value for
  // the block-size. If so, return the value, otherwise return -1.
  auto blockSizeX = launch.blockSizeX().getDefiningOp();
  if (blockSizeX->hasTrait<mlir::OpTrait::ConstantLike>()) {
    SmallVector<OpFoldResult, 1> foldResults;
    auto foldReturn = blockSizeX->fold({}, foldResults);
    if (failed(foldReturn)) {
      // Unable to fold the constant
      return -1;
    }
    if (auto constAttr = foldResults.front().dyn_cast<Attribute>()) {
      if (auto constIntAttr = constAttr.dyn_cast<IntegerAttr>()) {
        return constIntAttr.getInt();
      }
    }
  }
  return -1;
}

int hasConstantBlockSize(gpu::GPUFuncOp gpuFunc) {
  //
  // Check that all symbol-users of a GPUFunc are LaunchFunc operations from the GPU dialect
  // and all have the identical constant block-size.
  // Return the constant block-size if both criteria are met and -1 otherwise.
  auto callers = SymbolTable::getSymbolUses(gpuFunc, gpuFunc->getParentOfType<mlir::ModuleOp>());
  int blockSizeX = 0;
  for (auto& call : *callers) {
    if (auto launch = dyn_cast<gpu::LaunchFuncOp>(call.getUser())) {
      auto constantBlockSize = getConstantBlockSize(launch);
      if (!blockSizeX || blockSizeX == constantBlockSize) {
        blockSizeX = constantBlockSize;
      } else {
        // The constant block size for this launch differs from previously encountered launches.
        blockSizeX = -1;
      }
    } else {
      // Unknown type of user for the GPU function.
      blockSizeX = -1;
    }
  }
  return blockSizeX;
}

struct FuncSharedMemoryInsertion : public mlir::OpRewritePattern<gpu::GPUFuncOp> {

  using mlir::OpRewritePattern<gpu::GPUFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::GPUFuncOp gpuFunc, PatternRewriter& rewriter) const override {
    //
    // Try to pre-load input memories to enable coalesced access. If multiple indices per sample
    // are accessed for an input memory, this will lead to non-coalesced accesses to global memory,
    // as the same index for different samples is #numFeatures elements apart. To avoid that,
    // pre-load the part of the input for this block into shared/work-group memory.

    //
    // Currently, this transformation is only performed if the block-sizes at all invocation sites have
    // the same constant value. Check hat.
    auto constantBlockSize = hasConstantBlockSize(gpuFunc);
    if (constantBlockSize < 1) {
      return rewriter.notifyMatchFailure(gpuFunc, "Block size not constant, cannot insert shared memory");
    }

    // Compute dominance information to determine the block in which pre-load instructions should be inserted.
    DominanceInfo domInfo(gpuFunc);
    auto rootBlock = domInfo.getRootNode(&gpuFunc.body())->getBlock();

    // For each argument of the function which is a MemRef, check if it is eligible for
    // transformation. To be eligible, it needs to fulfill the following criteria:
    // 1. All uses inside the function must be SPNBatchReads
    // 2. There must be at least two SPNBatchRead with differing feature index.
    SmallVector<Value, 5> inputMemories;
    for (auto& arg : gpuFunc.body().getArguments()) {
      if (arg.getType().isa<MemRefType>()) {
        auto eligible = true;
        auto useCount = 0;
        llvm::DenseSet<unsigned> indices;
        for (auto U : arg.getUsers()) {
          // Only MemRefs that are only used by non-transposed
          // SPNBatchReads are considered eligible for this transformation.
          if (auto batchRead = dyn_cast<low::SPNBatchRead>(U)) {
            eligible &= !batchRead.transposed().getValueOr(false);
            if (!indices.count(batchRead.staticIndex())) {
              // Check that we have not encountered the same index before.
              ++useCount;
              indices.insert(batchRead.staticIndex());
            }
          } else {
            eligible &= false;
          }
        }
        if (eligible && useCount > 1) {
          inputMemories.push_back(arg);
        }
      }
    }

    // Skip this kernel if no argument is eligible for transformation.
    if (inputMemories.empty()) {
      return rewriter.notifyMatchFailure(gpuFunc, "No memories eligible for transformation found");
    }


    rewriter.startRootUpdate(gpuFunc);
    auto loc = gpuFunc->getLoc();
    auto maxSharedMem = CUDATargetInformation::maxSharedMemoryPerBlock(loc);
    for (auto inputMem : inputMemories) {

      // Collect all the SPNBatchRead using this MemRef. From the previous check above
      // we know that all users must be SPNBatchRead.
      SmallVector<low::SPNBatchRead, 10> reads;
      for (auto U : inputMem.getUsers()) {
        auto read = cast<low::SPNBatchRead>(U);
        assert(read);
        reads.push_back(read);
      }

      // Iterate through the BatchReads to determine the minimum &
      // maximum feature index and the number of features.
      unsigned minIndex = std::numeric_limits<unsigned>::max();
      unsigned maxIndex = std::numeric_limits<unsigned>::min();
      for (auto& read : reads) {
        minIndex = std::min(minIndex, read.staticIndex());
        maxIndex = std::max(maxIndex, read.staticIndex());
      }
      auto numFeatures = (maxIndex - minIndex) + 1;

      // Check that all input elements for all threads in the block fit into shared memory.
      auto elementType = inputMem.getType().cast<MemRefType>().getElementType();
      auto memRefType = MemRefType::get({numFeatures, constantBlockSize},
                                        elementType,
                                        {}, 3);
      auto memRefBytes = memRefType.getSizeInBits() / 8;
      if (memRefBytes > maxSharedMem) {
        gpuFunc.emitWarning() << "Cannot preload input, remaing shared memory of "
                              << maxSharedMem << " bytes is insufficient, "
                              << memRefBytes << " bytes required";
        continue;
      }
      maxSharedMem -= memRefBytes;

      // Create a shared memory (address space 3) with workgroup attribution, attached to this function.
      auto sharedMem = gpuFunc.addWorkgroupAttribution(memRefType);

      //
      // Preload from global memory into shared/workgroup memory.
      // To make the accesses to global memory coalesced, neighbouring threads will load neighbouring
      // features.
      // In each iteration of the created for-loop, the threads team up to load the inputs for **one**
      // input sample. By looping over multiple samples, we will load all inputs for all threads in the block.
      rewriter.setInsertionPoint(rootBlock->getTerminator());
      // Calculate the base-index in global memory from the block-ID and block-size.
      auto blockDim = rewriter.create<gpu::BlockDimOp>(loc, rewriter.getIndexType(),
                                                       rewriter.getStringAttr("x"));
      auto blockID = rewriter.create<gpu::BlockIdOp>(loc, rewriter.getIndexType(),
                                                     rewriter.getStringAttr("x"));
      auto baseIndex = rewriter.create<MulIOp>(loc, blockDim, blockID);
      auto threadID = rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(),
                                                       rewriter.getStringAttr("x"));
      auto maxFeature = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(maxIndex));
      // Insert a for-loop iterating from 0 to blockSize in steps of 1.
      auto constZero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
      auto constBlockSize = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(constantBlockSize));
      auto step = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
      auto loop = rewriter.create<scf::ForOp>(loc, constZero, constBlockSize, step);
      rewriter.setInsertionPointToStart(&loop.getLoopBody().front());
      auto sampleIndex = rewriter.create<AddIOp>(loc, baseIndex, loop.getInductionVar());
      // If the number of features exceeds the number of threads in the block, threads might need to
      // load multiple inputs.
      auto numRounds = llvm::divideCeil(numFeatures, constantBlockSize);
      for (unsigned i = 0; i < numRounds; ++i) {
        auto featureOffset = rewriter.create<ConstantOp>(loc,
                                                         rewriter.getIndexAttr(i * constantBlockSize + minIndex));
        auto featureIndex = rewriter.create<AddIOp>(loc, threadID, featureOffset);
        if (i == (numRounds - 1)) {
          // In the last round, we need to be careful to only load data in-bounds for the currently loaded,
          // single sample. Insert an if, so that only threads that load a valid feature index will participate
          // in the load.
          auto inBounds = rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, featureIndex, maxFeature);
          auto ifCheck = rewriter.create<scf::IfOp>(loc, inBounds, false);
          rewriter.setInsertionPointToStart(&ifCheck.thenRegion().front());
        }
        // Load from global memory, transpose and store into shared memory.
        auto readGlobal = rewriter.create<memref::LoadOp>(loc, inputMem, ValueRange{sampleIndex, featureIndex});
        auto sharedMemOffset = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i * constantBlockSize));
        auto sharedMemIndex = rewriter.create<AddIOp>(loc, threadID, sharedMemOffset);
        (void) rewriter.create<memref::StoreOp>(loc, readGlobal, sharedMem,
                                                ValueRange{sharedMemIndex, loop.getInductionVar()});
      }

      //
      // Process all SPNBatchRead that used the original global MemRef to instead load from the transposed shared mem.
      OwningRewritePatternList patterns(gpuFunc.getContext());
      patterns.insert<RewriteBatchReadtoSharedMem>(gpuFunc.getContext(), sharedMem, minIndex);
      mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
      for (auto& read : reads) {
        (void) applyOpPatternsAndFold(read, frozenPatterns);
      }
    }
    // Insert a barrier (__syncthreads()) to make sure all memory loading is finished before the
    // threads resume computation.
    rewriter.setInsertionPoint(rootBlock->getTerminator());
    rewriter.create<gpu::BarrierOp>(loc);
    rewriter.finalizeRootUpdate(gpuFunc);
    return mlir::success();
  }
};

void mlir::spn::LoSPNGPUSharedMemoryInsertionPass::runOnOperation() {
  auto module = getOperation();
  auto* context = &getContext();
  OwningRewritePatternList patterns(context);
  patterns.insert<FuncSharedMemoryInsertion>(context);
  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  // Apply the pattern to all GPUFuncs in the module.
  module->walk([&frozenPatterns](gpu::GPUFuncOp gpuFunc) {
    (void) applyOpPatternsAndFold(gpuFunc, frozenPatterns);
  });
}

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNGPUSharedMemoryInsertionPass() {
  return std::make_unique<LoSPNGPUSharedMemoryInsertionPass>();
}