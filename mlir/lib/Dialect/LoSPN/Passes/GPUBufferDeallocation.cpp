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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "LoSPNPassDetails.h"
#include "LoSPN/LoSPNPasses.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
  namespace spn {
    namespace low {

      namespace {

        class InsertionLocation {

        public:

          InsertionLocation(Operation* op, PostDominanceInfo* postDomInfo) : operation{op}, block{nullptr},
                                                                             domInfo{postDomInfo} {}

        private:

          InsertionLocation(Block* bloc, PostDominanceInfo* postDomInfo) : operation{nullptr}, block{bloc},
                                                                           domInfo{postDomInfo} {}

        public:

          InsertionLocation merge(Operation* other) const {
            assert(isOperation() ^ isBlock());
            if (this->isOperation()) {
              if (domInfo->postDominates(operation, other)) {
                return {operation, domInfo};
              }
              if (domInfo->postDominates(other, operation)) {
                return {other, domInfo};
              }
              auto commonBlock = domInfo->findNearestCommonDominator(operation->getBlock(), other->getBlock());
              return {commonBlock, domInfo};
            }
            assert(isBlock());
            if (block == other->getBlock()) {
              // So far, we assumed that we would insert at the beginning of the block. If the other operation
              // is contained in the same block, we can simply insert after that operation.
              return {other, domInfo};
            }
            auto commonBlock = domInfo->findNearestCommonDominator(block, other->getBlock());
            return {commonBlock, domInfo};
          }

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

      class DeallocateGPUBuffer : public OpRewritePattern<gpu::AllocOp> {

      public:

        DeallocateGPUBuffer(MLIRContext* ctx,
                            PostDominanceInfo* postDominanceInfo) : OpRewritePattern<gpu::AllocOp>(ctx, 1),
                                                                    domInfo{postDominanceInfo} {}

        LogicalResult matchAndRewrite(gpu::AllocOp alloc, PatternRewriter& rewriter) const override {
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
            return rewriter.notifyMatchFailure(alloc, "Buffer is already de-allocated");
          }
          rewriter.startRootUpdate(alloc);
          auto iterator = alloc.memref().getUsers().begin();
          InsertionLocation insertLoc{*iterator, domInfo};
          std::next(iterator);
          for (; iterator != alloc.memref().getUsers().end(); ++iterator) {
            insertLoc = insertLoc.merge(*iterator);
          }
          auto save = insertLoc.setInsertionPoint(rewriter);
          rewriter.create<gpu::DeallocOp>(alloc->getLoc(), llvm::None, ValueRange{}, alloc.memref());
          rewriter.restoreInsertionPoint(save);
          rewriter.finalizeRootUpdate(alloc);
          return success();
        }

      private:

        PostDominanceInfo* domInfo;

      };

      struct GPUBufferDeallocation : public GPUBufferDeallocationBase<GPUBufferDeallocation> {
      protected:
        void runOnOperation() override {
          auto func = getOperation();
          PostDominanceInfo domInfo{func};
          if (!domInfo.hasDominanceInfo(&func.body())) {
            signalPassFailure();
          }
          RewritePatternSet patterns(func.getContext());
          patterns.insert<DeallocateGPUBuffer>(func.getContext(), &domInfo);
          (void) mlir::applyPatternsAndFoldGreedily(func, FrozenRewritePatternSet(std::move(patterns)));
          func.dump();
        }
      };
    }
  }
}

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> mlir::spn::low::createGPUBufferDeallocationPass() {
  return std::make_unique<GPUBufferDeallocation>();
}