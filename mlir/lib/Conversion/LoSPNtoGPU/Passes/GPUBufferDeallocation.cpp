//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "LoSPNtoGPUPassDetails.h"
#include "LoSPNtoGPU/LoSPNtoGPUPasses.h"
#include "mlir/Dialect/GPU/GPUDialect.h"

namespace mlir {
  namespace spn {

    namespace {

      // Calculate the correct location to insert the de-allocation so it happens after all users
      // have completed computation. The de-allocation can either be inserted directly after the last user's
      // operation, in case that operation post-dominates all other users, or at the beginning of a block
      // post-dominating all users.
      //
      // NOTE: std::variant might be an interesting implementation alternative here after we switch to C++17.
      class InsertionLocation {

      public:

        InsertionLocation(Operation* op, PostDominanceInfo* postDomInfo) : operation{op}, block{nullptr},
                                                                           domInfo{postDomInfo} {
          assert(operation);
        }

      private:

        InsertionLocation(Block* bloc, PostDominanceInfo* postDomInfo) : operation{nullptr}, block{bloc},
                                                                         domInfo{postDomInfo} {
          assert(block);
        }

      public:

        /// Find the correct insertion location based on the existing and another user.
        /// \param other The other user.
        /// \return The location to perform the de-allocation.
        InsertionLocation merge(Operation* other) const {
          assert(isOperation() ^ isBlock());
          if (this->isOperation()) {
            // If either one of the two operations post-dominates the other,
            // it is safe to insert the de-allocation after that operation.
            if (domInfo->postDominates(operation, other)) {
              return {operation, domInfo};
            }
            if (domInfo->postDominates(other, operation)) {
              return {other, domInfo};
            }
            // In case none of the operations dominates the other, the de-allocation must be placed in
            // the block dominating both operations.
            auto commonBlock = domInfo->findNearestCommonDominator(operation->getBlock(), other->getBlock());
            return {commonBlock, domInfo};
          }
          assert(isBlock());
          if (block == other->getBlock()) {
            // So far, we assumed that we would insert at the beginning of the block. If the other operation
            // is contained in the same block, we can simply insert after that operation.
            return {other, domInfo};
          }
          // We need to find the block that dominates both blocks, which could also be one of the
          // two blocks.
          auto commonBlock = domInfo->findNearestCommonDominator(block, other->getBlock());
          return {commonBlock, domInfo};
        }

        /// Set the rewriter's insertion point to the correct location where the de-allocation
        /// should be inserted.
        /// \param rewriter The PatternRewriter.
        /// \return The saved insertion point **before** setting the rewriter.
        mlir::OpBuilder::InsertPoint setInsertionPoint(PatternRewriter& rewriter) const {
          auto save = rewriter.saveInsertionPoint();
          if (isOperation()) {
            rewriter.setInsertionPointAfter(operation);
          } else {
            rewriter.setInsertionPointToStart(block);
          }
          return save;
        }

      private:

        bool isOperation() const {
          return operation != nullptr;
        }

        bool isBlock() const {
          return block != nullptr;
        }

        Operation* operation;

        Block* block;

        PostDominanceInfo* domInfo;

      };

    }

    ///
    /// Simple pattern to insert a de-allocation for each allocation on the GPU.
    class DeallocateGPUBuffer : public OpRewritePattern<gpu::AllocOp> {

    public:

      DeallocateGPUBuffer(MLIRContext* ctx,
                          PostDominanceInfo* postDominanceInfo) : OpRewritePattern<gpu::AllocOp>(ctx, 1),
                                                                  domInfo{postDominanceInfo} {}

      LogicalResult matchAndRewrite(gpu::AllocOp alloc, PatternRewriter& rewriter) const override {
        // Check if any user of the memref is already a de-allocation.
        auto isDeallocated = llvm::any_of(alloc.memref().getUsers(), [&alloc](Operation* op) {
          if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
            SmallVector<MemoryEffects::EffectInstance> effects;
            memEffect.getEffectsOnValue(alloc.memref(), effects);
            return llvm::any_of(effects, [](MemoryEffects::EffectInstance effect) {
              return isa<MemoryEffects::Free>(effect.getEffect());
            });
          }
          return false;
        });
        if (isDeallocated) {
          // We assume that the existing de-allocation is correct, i.e., it correctly
          // post-dominates all other users of the memref.
          return rewriter.notifyMatchFailure(alloc, "Buffer is already de-allocated");
        }
        rewriter.startRootUpdate(alloc);
        // Calculate a safe location to insert the de-allocation so it correctly post-dominates
        // all users.
        auto iterator = alloc.memref().getUsers().begin();
        InsertionLocation insertLoc{*iterator, domInfo};
        std::next(iterator);
        for (; iterator != alloc.memref().getUsers().end(); ++iterator) {
          insertLoc = insertLoc.merge(*iterator);
        }
        auto save = insertLoc.setInsertionPoint(rewriter);
        // Insert the actual de-allocation.
        rewriter.create<gpu::DeallocOp>(alloc->getLoc(), std::nullopt, ValueRange{}, alloc.memref());
        rewriter.restoreInsertionPoint(save);
        rewriter.finalizeRootUpdate(alloc);
        return success();
      }

    private:

      PostDominanceInfo* domInfo;

    };

    ///
    /// Pass to insert de-allocation of GPU memory buffers after all users have completed.
    struct GPUBufferDeallocation : public GPUBufferDeallocationBase<GPUBufferDeallocation> {
    protected:
      void runOnOperation() override {
        auto func = getOperation();
        PostDominanceInfo domInfo{func};
        if (!domInfo.hasSSADominance(&func.body())) {
          signalPassFailure();
        }
        RewritePatternSet patterns(func.getContext());
        patterns.insert<DeallocateGPUBuffer>(func.getContext(), &domInfo);
        (void) mlir::applyPatternsAndFoldGreedily(func, FrozenRewritePatternSet(std::move(patterns)));
      }
    };
  }
}

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> mlir::spn::createGPUBufferDeallocationPass() {
  return std::make_unique<GPUBufferDeallocation>();
}