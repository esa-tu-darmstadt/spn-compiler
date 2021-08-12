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
#include "LoSPNtoGPUPassDetails.h"
#include "LoSPNtoGPU/LoSPNtoGPUPasses.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
  namespace spn {

    //
    // We can replace one device buffer with another, previously copied one in the
    // following scenario:
    //
    //  (1) copy(%host, %device1) or copy(%device1, %host)  <-- %device1 and %host are identical
    //      No writes to either %host or %device1
    //  (2) copy(%device2, %host)                           <-- %device1 and %device2 have the same content.
    //      No writes to either %device1 or %device2
    //  (3) gpu.launch(%device2){ no write to %device2 }    <-- %device1 can be used instead of %device2
    //
    // Using %device1 instead of %device2 in (3) has the potential to make the copy in (2) unnecessary, and
    // might even make a copy from device to host in (1) superfluous.
    //
    // Currently, we limit the replacement to a single (basic) block, in order to simplify dataflow analysis.

    struct BufferCopyPropagation : public OpRewritePattern<FuncOp> {

      using OpRewritePattern<FuncOp>::OpRewritePattern;

      LogicalResult matchAndRewrite(FuncOp op, PatternRewriter& rewriter) const override {
        llvm::dbgs() << "Applying pattern to function\n";
        auto changed = false;
        rewriter.startRootUpdate(op);
        for (auto& region : op->getRegions()) {
          for (auto& block : region.getBlocks()) {
            changed |= propagateLocal(block, rewriter);
          }
        }
        if (!changed) {
          rewriter.cancelRootUpdate(op);
          llvm::dbgs() << "Did not perform any replacements\n";
          return rewriter.notifyMatchFailure(op, "Did not perform any replacements");
        }
        rewriter.finalizeRootUpdate(op);
        return success();
      }

    private:

      bool propagateLocal(Block& block, PatternRewriter& rewriter) const {
        SmallVector<std::pair<Value, Value>> copies;
        SmallVector<std::pair<Value, Value>> identicalBuffers;
        bool replacementPerformed = false;
        for (auto& op : block.getOperations()) {
          if (auto gpuLaunch = dyn_cast<gpu::LaunchOp>(op)) {
            SmallVector<Value> replace;
            llvm::dbgs() << "Currently available identical buffers: " << identicalBuffers.size() << "\n";
            for (auto it = identicalBuffers.begin(); it != identicalBuffers.end();) {
              Value copy = it->first;
              Value original = it->second;
              if (potentiallyWrites(gpuLaunch, copy)) {
                llvm::dbgs() << "Potentially writes copy\n";
                killWritten(copies, copy);
                it = identicalBuffers.erase(it);
              } else if (potentiallyWrites(gpuLaunch, original)) {
                llvm::dbgs() << "Potentially writes original\n";
                killWritten(copies, original);
                it = identicalBuffers.erase(it);
              } else {
                auto numUsesBefore = std::distance(copy.getUses().begin(), copy.getUses().end());
                gpuLaunch->walk([&](Operation* op) {
                  op->replaceUsesOfWith(copy, original);
                });
                auto numUsesAfter = std::distance(copy.getUses().begin(), copy.getUses().end());
                if (numUsesAfter < numUsesBefore) {
                  replacementPerformed = true;
                  llvm::dbgs() << "Performed replacement\n";
                }
                ++it;
              }
            }
          } else if (auto copy = dyn_cast<gpu::MemcpyOp>(op)) {
            Value src = copy.src();
            Value dst = copy.dst();
            auto isHtoD = isGPUMemory(dst) && !isGPUMemory(src);
            auto isDtoH = !isGPUMemory(dst) && isGPUMemory(src);
            assert(isHtoD ^ isDtoH);
            if (isHtoD) {
              for (auto it = identicalBuffers.begin(); it != identicalBuffers.end();) {
                if (it->first == dst || it->second == dst) {
                  it = identicalBuffers.erase(it);
                } else {
                  ++it;
                }
              }
              for (auto p : copies) {
                if (p.first == src) {
                  llvm::dbgs() << p.second << " and " << dst << " are identical copies\n";
                  identicalBuffers.emplace_back(dst, p.second);
                }
              }
              copies.emplace_back(src, dst);
              llvm::dbgs() << dst << " is a copy of " << src << "\n";
            }
            if (isDtoH) {
              for (auto it = copies.begin(); it != copies.end();) {
                if (it->first == dst) {
                  it = copies.erase(it);
                } else {
                  ++it;
                }
              }
              copies.emplace_back(dst, src);
            }
          } else {
            for (Value operand : op.getOperands()) {
              if (potentiallyWrites(&op, operand)) {
                killWritten(copies, operand);
                killWritten(identicalBuffers, operand);
              }
            }
          }
        }
        return replacementPerformed;
      }

      // TODO Make static?
      bool potentiallyWrites(Operation* op, Value memRef) const {
        if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
          SmallVector<MemoryEffects::EffectInstance> effects;
          memEffect.getEffectsOnValue(memRef, effects);
          // Check if any of the effects is not read or allocate.
          return llvm::any_of(effects, [](MemoryEffects::EffectInstance effect) {
            return !isa<MemoryEffects::Read>(effect.getEffect()) && !isa<MemoryEffects::Allocate>(effect.getEffect());
          });
        }
        if (auto gpuLaunch = dyn_cast<gpu::LaunchOp>(op)) {
          auto walkResult = gpuLaunch.body().walk([this, &memRef](Operation* op) {
            if (potentiallyWrites(op, memRef)) {
              llvm::dbgs() << *op << " potentially writes " << memRef << "\n";
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
          return walkResult.wasInterrupted();
        }
        // Conservatively assume a user with no side effect interface writes.
        return (llvm::find(memRef.getUsers(), op) != memRef.getUsers().end());
      }

      void killWritten(llvm::SmallVectorImpl<std::pair<Value, Value>>& gen, Value& written) const {
        for (auto it = gen.begin(); it != gen.end();) {
          auto& pair = *it;
          auto kill = (written == pair.first) || (written = pair.second);
          if (kill) {
            llvm::dbgs() << "Killing (" << pair.first << ", " << pair.second << ")\n";
            it = gen.erase(it);
          } else {
            ++it;
          }
        }
      }

      bool isGPUMemory(Value memRef) const {
        assert(memRef.getType().isa<MemRefType>());
        auto op = memRef.getDefiningOp();
        return op && isa<gpu::AllocOp>(op);
      }
    };

    struct CopyOpElimination : public OpRewritePattern<FuncOp> {

      using OpRewritePattern<FuncOp>::OpRewritePattern;

      LogicalResult matchAndRewrite(FuncOp op, PatternRewriter& rewriter) const override {
        PostDominanceInfo domInfo(op);
        auto changed = false;
        rewriter.startRootUpdate(op);
        // Copies to the function arguments (i.e., the block arguments of the entry block)
        // must not be eliminated, as they might be used outside the function.
        llvm::SmallPtrSet<Value, 10> funcArgs;
        funcArgs.insert(op.body().front().args_begin(), op.body().front().args_end());
        for (auto& region : op->getRegions()) {
          if (domInfo.hasDominanceInfo(&region)) {
            // Skip regions for which no dominance info is available.
            for (auto& block : region.getBlocks()) {
              changed |= eliminateLocal(block, rewriter, domInfo, funcArgs);
            }
          }
        }
        if (!changed) {
          rewriter.cancelRootUpdate(op);
          llvm::dbgs() << "Did not eliminate any data transfers\n";
          return rewriter.notifyMatchFailure(op, "Did not eliminate any data transfers");
        }
        rewriter.finalizeRootUpdate(op);
        return success();
      }

      static bool eliminateLocal(Block& block, PatternRewriter& rewriter, PostDominanceInfo& domInfo,
                                 llvm::SmallPtrSetImpl<Value>& funcArgs) {
        bool changed = false;
        SmallPtrSet<Operation*, 10> eliminated;
        for (auto it = block.getOperations().rbegin(); it != block.getOperations().rend(); ++it) {
          if (auto copy = dyn_cast<gpu::MemcpyOp>(*it)) {
            auto canBeEliminated = !funcArgs.contains(copy.dst()) &&
                llvm::all_of(copy.dst().getUsers(), [&](Operation* op) {
                  return eliminated.contains(op) || domInfo.postDominates(copy, op) || writes(op, copy.dst());
                });
            if (canBeEliminated) {
              llvm::dbgs() << "Eliminating copy " << copy << "\n";
              eliminated.insert(copy);
              changed = true;
            }
          }
        }
        for (auto* op : eliminated) {
          rewriter.eraseOp(op);
        }
        return changed;
      }

      static bool writes(Operation* op, Value memRef) {
        if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
          SmallVector<MemoryEffects::EffectInstance> effects;
          memEffect.getEffectsOnValue(memRef, effects);
          return llvm::all_of(effects, [](MemoryEffects::EffectInstance effect) {
            return isa<MemoryEffects::Write>(effect.getEffect());
          });
        }
        if (auto gpuLaunch = dyn_cast<gpu::LaunchOp>(op)) {
          auto walkResult = gpuLaunch.body().walk([&memRef](Operation* op) {
            if (!writes(op, memRef)) {
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
          return walkResult.wasInterrupted();
        }
        return false;
      }

    };

    struct GPUCopyEliminationPass : public GPUCopyEliminationBase<GPUCopyEliminationPass> {

    protected:
      void runOnOperation() override {
        llvm::dbgs() << "Running on function\n";
        auto module = getOperation();
        RewritePatternSet patterns(module.getContext());
        patterns.insert<BufferCopyPropagation>(module.getContext(), 2);
        patterns.insert<CopyOpElimination>(module.getContext());
        (void) mlir::applyPatternsAndFoldGreedily(module, FrozenRewritePatternSet(std::move(patterns)));
        module.dump();
        //assert(false);
      }

    };

  }
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::spn::createGPUCopyEliminationPass() {
  return std::make_unique<GPUCopyEliminationPass>();
}