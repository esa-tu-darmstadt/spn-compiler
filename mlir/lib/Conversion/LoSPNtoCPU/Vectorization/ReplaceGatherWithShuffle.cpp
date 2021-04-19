//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/VectorOptimizationPasses.h"
#include "mlir/Rewrite/FrozenRewritePatternList.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "llvm/ADT/IndexedMap.h"

using namespace mlir;
using namespace mlir::spn;

class ReplaceBatchReadWithShuffle : public OpRewritePattern<low::SPNBatchRead> {

public:

  explicit ReplaceBatchReadWithShuffle(MLIRContext* ctx, llvm::IndexedMap<Value>& replacements) :
      OpRewritePattern<low::SPNBatchRead>(ctx, 1), replace{replacements} {}

  LogicalResult matchAndRewrite(low::SPNBatchRead op, PatternRewriter& rewriter) const override {
    //
    // The transformation loads the input values with regular loads (instead of gathers)
    // and transposes a tile of the input using shuffles.
    // The BatchRead does not need to read anything add more, but can just use the
    // transposed/shuffled vector.
    if (!replace.inBounds(op.sampleIndex()) || !replace[op.sampleIndex()]) {
      // No replacement found.
      return rewriter.notifyMatchFailure(op, "No replacement defined");
    }
    auto shuffled = replace[op.sampleIndex()];
    rewriter.replaceOpWithNewOp<low::SPNConvertToScalar>(op, shuffled.getType().cast<VectorType>().getElementType(),
                                                         shuffled);
    return mlir::success();
  }

private:

  llvm::IndexedMap<Value>& replace;

};

/*
 * Bunch of helper functions declared in an anonymous namespace.
 */
namespace {

  std::pair<mlir::Value, mlir::Value> combineHalves(mlir::Value& leftIn, mlir::Value& rightIn,
                                                    unsigned vectorSize, unsigned step,
                                                    mlir::PatternRewriter& rewriter,
                                                    mlir::Location loc) {
    unsigned leftIndex = 0;
    unsigned rightIndex = vectorSize;
    llvm::SmallVector<int64_t, 8> firstPermutation;
    for (unsigned i = 0; i < vectorSize / (step * 2); ++i) {
      for (unsigned k = 0; k < step; ++k) {
        firstPermutation.push_back(leftIndex++);
      }
      for (unsigned k = 0; k < step; ++k) {
        firstPermutation.push_back(rightIndex++);
      }
    }
    llvm::SmallVector<int64_t, 8> secondPermutation;
    for (unsigned i = 0; i < vectorSize / (step * 2); ++i) {
      for (unsigned k = 0; k < step; ++k) {
        secondPermutation.push_back(leftIndex++);
      }
      for (unsigned k = 0; k < step; ++k) {
        secondPermutation.push_back(rightIndex++);
      }
    }

    auto leftPermutation = rewriter.create<mlir::vector::ShuffleOp>(loc, leftIn, rightIn, firstPermutation);
    auto rightPermutation = rewriter.create<mlir::vector::ShuffleOp>(loc, leftIn, rightIn, secondPermutation);
    return {leftPermutation, rightPermutation};
  }

  llvm::SmallVector<mlir::Value, 8> transposeByPermutation(llvm::ArrayRef<mlir::Value> loadedVectors,
                                                           unsigned vectorSize,
                                                           mlir::PatternRewriter& rewriter,
                                                           mlir::Location loc) {
    llvm::SmallVector<mlir::Value, 8> vectors;
    for (auto v : loadedVectors) {
      vectors.push_back(v);
    }
    unsigned numPermutationStage = log2(vectorSize);
    for (unsigned i = 0; i < numPermutationStage; ++i) {
      unsigned distance = pow(2, i);
      llvm::SmallVector<mlir::Value, 8> newVectors;
      unsigned index = 0;
      for (unsigned j = 0; j < vectorSize / (distance * 2); ++j) {
        // Work on some elements
        for (unsigned k = 0; k < distance; ++k) {
          auto leftIn = vectors[index];
          auto rightIn = vectors[index + distance];
          auto outVec = combineHalves(leftIn, rightIn, vectorSize, distance, rewriter, loc);
          newVectors.push_back(outVec.first);
          newVectors.push_back(outVec.second);
          ++index;
        }
        // Skip some elements
        index += distance;
      }
      vectors = std::move(newVectors);
    }
    return vectors;
  }

}

struct FuncReplaceGatherWithShuffle : public OpRewritePattern<FuncOp> {

  using OpRewritePattern<FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp func, PatternRewriter& rewriter) const override {
    //
    // Instead of using expensive gather + insert operations to load the same feature for multiple
    // samples into a vector, this transformation will load a vectorWidth x vectorWidth block
    // of input values using ordinary loads. The block will contain vectorWidth features for
    // vectorWidth many samples. To have one feature for multiple samples in one vector,
    // the block is transposed using vector shuffle operations.

    // For each argument of the function which is a MemRef, check if it is eligible for
    // transformation. To be eligible, there must be at least as many vectorized
    // SPNBatchRead using the memref as the vector width.
    SmallVector<Value, 5> inputMemories;
    for (auto& arg : func.body().getArguments()) {
      if (arg.getType().isa<MemRefType>()) {
        auto eligible = true;
        auto useCount = 0;
        unsigned minVectorWidth = std::numeric_limits<unsigned>::max();
        llvm::DenseSet<unsigned> indices;
        for (auto U : arg.getUsers()) {
          if (auto batchRead = dyn_cast<low::SPNBatchRead>(U)) {
            eligible &= true;
            if (batchRead.checkVectorized() && !indices.count(batchRead.sampleIndex())) {
              // Check that we have not encountered the same index before.
              ++useCount;
              minVectorWidth = std::min(minVectorWidth, batchRead.vectorFactor());
              indices.insert(batchRead.sampleIndex());
            }
          } else if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(U)) {
            SmallVector<MemoryEffects::EffectInstance, 1> effects;
            memEffect.getEffectsOnValue(arg, effects);
            for (auto e : effects) {
              if (isa<MemoryEffects::Write>(e.getEffect())) {
                eligible &= false;
              }
            }
          }
        }
        if (eligible && useCount >= minVectorWidth) {
          inputMemories.push_back(arg);
        }
      }
    }

    // Skip this kernel if no argument is eligible for transformation.
    if (inputMemories.empty()) {
      return rewriter.notifyMatchFailure(func, "No memories eligible for transformation found");
    }

    rewriter.startRootUpdate(func);
    auto loc = func->getLoc();
    bool changed = false;
    for (auto inputMem : inputMemories) {
      // Collect all the SPNBatchRead using this MemRef.
      SmallVector<low::SPNBatchRead, 10> reads;
      llvm::IndexedMap<bool> indices;
      for (auto U : inputMem.getUsers()) {
        if (auto read = dyn_cast<low::SPNBatchRead>(U)) {
          if (read.checkVectorized()) {
            // Collect only vectorized BatchRead for this transformation.
            reads.push_back(read);
            indices.grow(read.sampleIndex());
            indices[read.sampleIndex()] = true;
          }
        }
      }

      // For simplification, we currently only perform this transformation if all vectorized reads
      // are (1) in the same block and (2) have the same vector-width.
      unsigned vector_width = reads.front().vectorFactor();
      Block* block = reads.front()->getBlock();
      DominanceInfo domInfo(func);
      low::SPNBatchRead firstRead = reads.front();
      bool failed = false;
      unsigned maxIndex = 0;
      for (auto& r : reads) {
        failed |= (vector_width != r.vectorFactor());
        failed |= (r->getBlock() != block);
        if (domInfo.dominates(r.getOperation(), firstRead)) {
          // Check for dominance to determine the first read.
          firstRead = r;
        }
        maxIndex = std::max(maxIndex, r.sampleIndex());
      }
      if (failed) {
        emitWarning(loc, Twine("Cannot replace gather with shuffle for input memory "));
        continue;
      }

      auto batchIndex = firstRead.batchIndex();
      auto vectorType = VectorType::get({vector_width}, inputMem.getType().cast<MemRefType>().getElementType());
      llvm::IndexedMap<Value> replacements;
      replacements.grow(maxIndex);
      for (unsigned i = 0; i < maxIndex; i += vector_width) {
        bool allPresent = true;
        // Check that at least one use of all indices in the current range is present.
        for (unsigned vectorIndex = i; vectorIndex < i + vector_width; ++vectorIndex) {
          allPresent &= indices.inBounds(vectorIndex) && indices[vectorIndex];
        }
        if (!allPresent) {
          continue;
        }
        changed = true;
        // Load the features [i, i + vector_width) for the samples [batchIndex, batchIndex + vectorWidth)
        rewriter.setInsertionPoint(firstRead);
        auto featureIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
        SmallVector<Value, 8> inputs;
        for (unsigned k = 0; k < vector_width; ++k) {
          auto sampleOffset = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(k));
          auto sampleIndex = rewriter.create<AddIOp>(loc, batchIndex, sampleOffset);
          auto inputVector = rewriter.create<vector::TransferReadOp>(loc, vectorType,
                                                                     inputMem,
                                                                     ValueRange{sampleIndex, featureIndex});
          inputs.push_back(inputVector);
        }

        auto transposed = transposeByPermutation(inputs, vector_width, rewriter, loc);
        for (unsigned k = 0; k < vector_width; ++k) {
          replacements[i + k] = transposed[k];
        }
      }
      //
      // Replace all SPNBatchRead that used the original MemRef with the shuffled vectors.
      OwningRewritePatternList patterns;
      patterns.insert<ReplaceBatchReadWithShuffle>(func.getContext(), replacements);
      mlir::FrozenRewritePatternList frozenPatterns(std::move(patterns));
      for (auto& read : reads) {
        applyOpPatternsAndFold(read, frozenPatterns);
      }
    }
    rewriter.finalizeRootUpdate(func);
    return mlir::success(changed);
  }

};

void ReplaceGatherWithShufflePass::runOnOperation() {
  auto module = getOperation();
  auto* context = &getContext();
  OwningRewritePatternList patterns;
  patterns.insert<FuncReplaceGatherWithShuffle>(context);
  mlir::FrozenRewritePatternList frozenPatterns(std::move(patterns));
  // Apply the pattern to all GPUFuncs in the module.
  module->walk([&frozenPatterns](FuncOp func) {
    applyOpPatternsAndFold(func, frozenPatterns);
  });
}

std::unique_ptr<mlir::Pass> mlir::spn::createReplaceGatherWithShufflePass() {
  return std::make_unique<ReplaceGatherWithShufflePass>();
}
