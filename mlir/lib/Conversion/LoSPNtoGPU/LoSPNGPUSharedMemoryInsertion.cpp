//
// This file is part of the SPNC project.
// Copyright (c) 2021 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/Rewrite/FrozenRewritePatternList.h"
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

using namespace mlir;
using namespace mlir::spn;

class RewriteBatchReadtoSharedMem : public mlir::OpRewritePattern<low::SPNBatchRead> {

public:

  RewriteBatchReadtoSharedMem(mlir::MLIRContext* ctx, Value sharedMemory) : OpRewritePattern{ctx, 1},
                                                                            sharedMem{sharedMemory} {}

  LogicalResult matchAndRewrite(low::SPNBatchRead op, PatternRewriter& rewriter) const override {
    auto loc = op->getLoc();
    auto threadID = rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(),
                                                     rewriter.getStringAttr("x"));
    auto sampleIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(op.sampleIndex()));
    rewriter.replaceOpWithNewOp<LoadOp>(op, sharedMem, ValueRange{sampleIndex, threadID});
    return success();
  }

private:

  Value sharedMem;

};

int getConstantBlockSize(gpu::LaunchFuncOp launch) {
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
  auto callers = SymbolTable::getSymbolUses(gpuFunc, gpuFunc->getParentOfType<mlir::ModuleOp>());
  int blockSizeX = 0;
  for (auto& call : *callers) {
    llvm::dbgs() << "Caller of gpu func: \n";
    call.getUser()->dump();
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
    auto constantBlockSize = hasConstantBlockSize(gpuFunc);
    if (constantBlockSize < 1) {
      return rewriter.notifyMatchFailure(gpuFunc, "Block size not constant, cannot insert shared memory");
    }
    llvm::dbgs() << gpuFunc.getName() << " is called with constant block-size\n";

    DominanceInfo domInfo(gpuFunc);
    auto rootBlock = domInfo.getRootNode(&gpuFunc.body())->getBlock();

    SmallVector<Value, 5> inputMemories;
    for (auto& arg : gpuFunc.body().getArguments()) {
      if (arg.getType().isa<MemRefType>()) {
        auto eligible = true;
        auto useCount = 0;
        for (auto U : arg.getUsers()) {
          // Only MemRefs that are only used by
          // SPNBatchReads are considered eligible for this transformation.
          eligible &= isa<low::SPNBatchRead>(U);
          ++useCount;
        }
        if (eligible && useCount > 1) {
          inputMemories.push_back(arg);
        }
      }
    }

    if (inputMemories.empty()) {
      return rewriter.notifyMatchFailure(gpuFunc, "No memories eligible for transformation found");
    }

    rewriter.startRootUpdate(gpuFunc);
    auto loc = gpuFunc->getLoc();
    for (auto inputMem : inputMemories) {
      SmallVector<low::SPNBatchRead, 10> reads;
      for (auto U : inputMem.getUsers()) {
        auto read = cast<low::SPNBatchRead>(U);
        assert(read);
        reads.push_back(read);
      }

      unsigned minIndex = std::numeric_limits<unsigned>::max();
      unsigned maxIndex = std::numeric_limits<unsigned>::min();
      for (auto& read : reads) {
        minIndex = std::min(minIndex, read.sampleIndex());
        maxIndex = std::max(maxIndex, read.sampleIndex());
      }
      auto numFeatures = (maxIndex - minIndex) + 1;
      auto memRefType = MemRefType::get({numFeatures, constantBlockSize},
                                        inputMem.getType().cast<MemRefType>().getElementType(),
                                        {}, 3);
      auto sharedMem = gpuFunc.addWorkgroupAttribution(memRefType);
      rewriter.setInsertionPoint(rootBlock->getTerminator());
      auto blockDim = rewriter.create<gpu::BlockDimOp>(loc, rewriter.getIndexType(),
                                                       rewriter.getStringAttr("x"));
      auto blockID = rewriter.create<gpu::BlockIdOp>(loc, rewriter.getIndexType(),
                                                     rewriter.getStringAttr("x"));
      auto baseIndex = rewriter.create<MulIOp>(loc, blockDim, blockID);
      auto constZero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
      auto constBlockSize = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(constantBlockSize));
      auto step = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
      auto threadID = rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(),
                                                       rewriter.getStringAttr("x"));
      auto maxFeature = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(maxIndex));
      auto loop = rewriter.create<scf::ForOp>(loc, constZero, constBlockSize, step);
      rewriter.setInsertionPointToStart(&loop.getLoopBody().front());
      auto sampleIndex = rewriter.create<AddIOp>(loc, baseIndex, loop.getInductionVar());
      auto numRounds = llvm::divideCeil(numFeatures, constantBlockSize);
      for (unsigned i = 0; i < numRounds; ++i) {
        auto featureOffset = rewriter.create<ConstantOp>(loc,
                                                         rewriter.getIndexAttr(i * constantBlockSize + minIndex));
        auto featureIndex = rewriter.create<AddIOp>(loc, threadID, featureOffset);
        if (i == (numRounds - 1)) {
          auto inBounds = rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, featureIndex, maxFeature);
          auto ifCheck = rewriter.create<scf::IfOp>(loc, inBounds, false);
          rewriter.setInsertionPointToStart(&ifCheck.thenRegion().front());
        }
        auto readGlobal = rewriter.create<LoadOp>(loc, inputMem, ValueRange{sampleIndex, featureIndex});
        auto storeShared = rewriter.create<StoreOp>(loc, readGlobal, sharedMem, ValueRange{featureIndex, sampleIndex});
      }

      OwningRewritePatternList patterns;
      patterns.insert<RewriteBatchReadtoSharedMem>(gpuFunc.getContext(), sharedMem);
      mlir::FrozenRewritePatternList frozenPatterns(std::move(patterns));
      for (auto& read : reads) {
        applyOpPatternsAndFold(read, frozenPatterns);
      }
    }
    rewriter.setInsertionPoint(rootBlock->getTerminator());
    rewriter.create<gpu::BarrierOp>(loc);
    gpuFunc->dump();
    rewriter.finalizeRootUpdate(gpuFunc);
    return mlir::success();
  }
};

void mlir::spn::LoSPNGPUSharedMemoryInsertionPass::runOnOperation() {
  auto module = getOperation();
  auto* context = &getContext();
  OwningRewritePatternList patterns;
  patterns.insert<FuncSharedMemoryInsertion>(context);
  mlir::FrozenRewritePatternList frozenPatterns(std::move(patterns));
  // TODO
  module->walk([&frozenPatterns](gpu::GPUFuncOp gpuFunc) {
    applyOpPatternsAndFold(gpuFunc, frozenPatterns);
  });
}

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNGPUSharedMemoryInsertionPass() {
  return std::make_unique<LoSPNGPUSharedMemoryInsertionPass>();
}