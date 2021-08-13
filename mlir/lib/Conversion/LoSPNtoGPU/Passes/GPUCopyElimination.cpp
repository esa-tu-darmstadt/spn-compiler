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

      class GPUKernelAnalysis {

      public:

        explicit GPUKernelAnalysis(gpu::LaunchOp launch) {
          launch.body().walk([this](Operation* op) {
            for (auto operand : op->getOperands()) {
              if (operand.getType().isa<MemRefType>()) {
                if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
                  SmallVector<MemoryEffects::EffectInstance> effects;
                  memEffect.getEffectsOnValue(operand, effects);
                  auto reads = llvm::any_of(effects, [](MemoryEffects::EffectInstance effect) {
                    return isa<MemoryEffects::Read>(effect.getEffect());
                  });
                  auto writes = llvm::any_of(effects, [](MemoryEffects::EffectInstance effect) {
                    return isa<MemoryEffects::Write>(effect.getEffect());
                  });
                  if (reads) {
                    read.insert(operand);
                    addToReaders(operand, op);
                  }
                  if (writes) {
                    written.insert(operand);
                  }
                } else {
                  // Conservatively assume users without side effect interface write
                  written.insert(operand);
                }
              }
            }
          });
        }

        llvm::SmallPtrSetImpl<Value>& readBuffers() {
          return read;
        }

        llvm::SmallPtrSetImpl<Value>& writeBuffers() {
          return written;
        }

        llvm::SmallVectorImpl<Operation*>& readersOf(Value memRef) {
          assert(readers.count(memRef));
          return *(readers[memRef]);
        }

      private:

        llvm::SmallPtrSet<Value, 10> read;
        llvm::SmallPtrSet<Value, 10> written;

        llvm::DenseMap<Value, std::unique_ptr<llvm::SmallVectorImpl<Operation*>>> readers;

        void addToReaders(Value memRef, Operation* reader) {
          if (!readers.count(memRef)) {
            readers.insert({memRef, std::make_unique<llvm::SmallVector<Operation*, 10>>()});
          }
          readers[memRef]->push_back(reader);
        }

      };

    }

    ///
    /// Pattern to replace GPU buffers with other GPU buffers that are already present on the device,
    /// e.g., for intermediate results.
    /// The replacement is currently only performed locally, i.e., within the borders of a block to avoid
    /// complicated data-flow analysis.
    struct BufferCopyPropagation : public OpRewritePattern<FuncOp> {

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

      using OpRewritePattern<FuncOp>::OpRewritePattern;

      LogicalResult matchAndRewrite(FuncOp op, PatternRewriter& rewriter) const override {
        auto changed = false;
        rewriter.startRootUpdate(op);
        // Iterate over all blocks (inside each region) and perform the transformation
        // locally inside each block.
        for (auto& region : op->getRegions()) {
          for (auto& block : region.getBlocks()) {
            changed |= propagateLocal(block, rewriter);
          }
        }
        if (!changed) {
          rewriter.cancelRootUpdate(op);
          return rewriter.notifyMatchFailure(op, "Did not perform any replacements");
        }
        rewriter.finalizeRootUpdate(op);
        return success();
      }

    private:

      bool propagateLocal(Block& block, PatternRewriter& rewriter) const {
        // Contains a pair (%host, %device) if the content of %host and %device is identical
        // due to a copy from one to the other.
        SmallVector<std::pair<Value, Value>> copies;
        // Contains a pair (%device1, %device2) if the contents of %device1 and %device2 is identical
        // after %device1 has been copied to/from a host-buffer, which was then copied to %device2.
        SmallVector<std::pair<Value, Value>> identicalBuffers;
        bool replacementPerformed = false;
        for (auto& op : block.getOperations()) {
          if (auto gpuLaunch = dyn_cast<gpu::LaunchOp>(op)) {
            GPUKernelAnalysis analysis{gpuLaunch};
            auto& written = analysis.writeBuffers();
            auto& read = analysis.readBuffers();
            for (auto it = identicalBuffers.begin(); it != identicalBuffers.end();) {
              Value copy = it->first;
              Value original = it->second;
              if (written.contains(copy)) {
                // We cannot perform the transformation, if this GPU kernel writes %device2, because using
                // %device1 instead of %device2 would then alter the contents of %device1.
                it = identicalBuffers.erase(it);
                // We also need to kill any pair (%host, %device2), because the content of %device2 will not
                // be identical to %host after this kernel has executed.
                killWritten(copies, copy);
              } else if (written.contains(original)) {
                // We cannot perform the transformation, if this GPU kernel writes %device1, because using
                // %device1 instead of %device2 would then alter the contents of %device2.
                it = identicalBuffers.erase(it);
                // We also need to kill any pair (%host, %device1), because the content of %device1 will not
                // be identical to %host after this kernel has executed.
                killWritten(copies, original);
              } else {
                if (read.contains(copy)) {
                  // It is legal to use %device1 (which might already be present on the GPU) instead of %device2,
                  // so we replace any use.
                  auto numUsesBefore = std::distance(copy.getUses().begin(), copy.getUses().end());
                  for (auto reader: analysis.readersOf(copy)) {
                    reader->replaceUsesOfWith(copy, original);
                  }
                  replacementPerformed = true;
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
              // We need to delete any pair (%device1, %device2) if this copy operation writes to
              // either %device1 or %device2, as their contents will not be identical after copying.
              for (auto it = identicalBuffers.begin(); it != identicalBuffers.end();) {
                if (it->first == dst || it->second == dst) {
                  it = identicalBuffers.erase(it);
                } else {
                  ++it;
                }
              }
              for (auto p : copies) {
                if (p.first == src) {
                  // If a pair (%host, %device1) reaches this copy of %host to %device, this means
                  // that the contents of %device1 and %device2 will be identical after this copy.
                  identicalBuffers.emplace_back(dst, p.second);
                }
              }
              // Add a pair (%host, %device) for this copy.
              copies.emplace_back(src, dst);
            }
            if (isDtoH) {
              // We need to delete any pair (%host, %device1) if we write to %host,
              // as the contents will not be identical after copying %device2 to %host
              // in this operation.
              for (auto it = copies.begin(); it != copies.end();) {
                if (it->first == dst) {
                  it = copies.erase(it);
                } else {
                  ++it;
                }
              }
              // Add a pair (%host, %device) for this copy.
              copies.emplace_back(dst, src);
            }
          } else {
            // For any other operation, we need to check whether it writes any of the existing host- or
            // device buffers and delete any pairing.
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

      /// Check if a operation potentially writes to a buffer.
      /// \param op The operation.
      /// \param memRef The buffer memref.
      /// \return True if op potentially writes to memRef.
      static bool potentiallyWrites(Operation* op, Value memRef) {
        // Try to use the SideEffect interfaces to determine behavior of op.
        if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
          SmallVector<MemoryEffects::EffectInstance> effects;
          memEffect.getEffectsOnValue(memRef, effects);
          // Check if any of the effects is not read or allocate.
          return llvm::any_of(effects, [](MemoryEffects::EffectInstance effect) {
            return !isa<MemoryEffects::Read>(effect.getEffect()) && !isa<MemoryEffects::Allocate>(effect.getEffect());
          });
        }
        // Conservatively assume a user with no side effect interface writes.
        return (llvm::find(memRef.getUsers(), op) != memRef.getUsers().end());
      }

      /// Kill (= remove) any pair that is affected by a write to written.
      /// \param gen Collection of pairs.
      /// \param written Memref.
      static void killWritten(llvm::SmallVectorImpl<std::pair<Value, Value>>& gen, Value& written) {
        for (auto it = gen.begin(); it != gen.end();) {
          auto& pair = *it;
          auto kill = (written == pair.first) || (written = pair.second);
          if (kill) {
            it = gen.erase(it);
          } else {
            ++it;
          }
        }
      }

      static bool isGPUMemory(Value memRef) {
        assert(memRef.getType().isa<MemRefType>());
        auto op = memRef.getDefiningOp();
        return op && isa<gpu::AllocOp>(op);
      }
    };

    // Pattern to eliminate unnecessary copies of buffers between GPU and host.
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
          // Skip regions for which no dominance info is available.
          if (domInfo.hasDominanceInfo(&region)) {
            for (auto& block : region.getBlocks()) {
              changed |= eliminateLocal(block, rewriter, domInfo, funcArgs);
            }
          }
        }
        if (!changed) {
          rewriter.cancelRootUpdate(op);
          return rewriter.notifyMatchFailure(op, "Did not eliminate any data transfers");
        }
        rewriter.finalizeRootUpdate(op);
        return success();
      }

      /// Eliminate unnecessary copies in a block.
      /// \param block Block.
      /// \param rewriter PatternRewriter.
      /// \param domInfo Post-dominance information to assess legality of the elimination.
      /// \param funcArgs Arguments of the containing function's entry blocks. Copies to function arguments
      ///                 must not be eliminated.
      /// \return True if the any copies were eliminated.
      static bool eliminateLocal(Block& block, PatternRewriter& rewriter, PostDominanceInfo& domInfo,
                                 llvm::SmallPtrSetImpl<Value>& funcArgs) {
        SmallPtrSet<Operation*, 10> eliminated;
        // Iterate the block in reverse order, because eliminating a copy further back might
        // make an earlier copy unnecessary, too.
        for (auto it = block.getOperations().rbegin(); it != block.getOperations().rend(); ++it) {
          if (auto copy = dyn_cast<gpu::MemcpyOp>(*it)) {
            // A copy can be eliminated iff
            // (a) it does not copy to a function argument (that might be used as an out-arg).
            // (b) no read from the destination buffer is performed after the copy, i.e., every other
            //     use is either post-dominated by the copy or only performs a write operation.
            //     This is pessimistically, as it does not take intermediate writes into account if any
            //     read occurs after the copy, which is necessary, as the write might only override parts of
            //     the copied content.
            auto canBeEliminated = !funcArgs.contains(copy.dst()) &&
                llvm::all_of(copy.dst().getUsers(), [&](Operation* op) {
                  return eliminated.contains(op) || domInfo.postDominates(copy, op) || writes(op, copy.dst());
                });
            if (canBeEliminated) {
              eliminated.insert(copy);
            }
          }
        }
        for (auto* op : eliminated) {
          rewriter.eraseOp(op);
        }
        return !eliminated.empty();
      }

      /// Check if an operation only writes to a buffer (i.e., does not read or perform any other operations)
      /// \param op The operation.
      /// \param memRef The buffer memref.
      /// \return True if op only performs write operations on memRef.
      static bool writes(Operation* op, Value memRef) {
        // Try to use the SideEffect interfaces to determine behavior of op.
        if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
          SmallVector<MemoryEffects::EffectInstance> effects;
          memEffect.getEffectsOnValue(memRef, effects);
          return llvm::all_of(effects, [](MemoryEffects::EffectInstance effect) {
            return isa<MemoryEffects::Write>(effect.getEffect());
          });
        }
        // For a GPU kernel, analyze the body of the kernel to find potential writes.
        if (auto gpuLaunch = dyn_cast<gpu::LaunchOp>(op)) {
          auto walkResult = gpuLaunch.body().walk([&memRef](Operation* op) {
            if (!writes(op, memRef)) {
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
          return walkResult.wasInterrupted();
        }
        // Conservatively assume the operation does not only write to memRef.
        // NOTE: A check if the operation is an actual user of memRef is not necessary here, as this function is
        // currently only applied to users of memRef.
        return false;
      }

    };

    ///
    /// Pass to eliminate unnecessary copy operations between host and GPU buffers.
    /// Tries to re-use intermediate result buffers already present on the GPU instead of copying intermediate
    /// results to the host and back to the GPU and removes any copy operation unnecessary after that transformation.
    struct GPUCopyEliminationPass : public GPUCopyEliminationBase<GPUCopyEliminationPass> {

    protected:
      void runOnOperation() override {
        auto module = getOperation();
        RewritePatternSet patterns(module.getContext());
        // Give the pattern for re-use of existing buffers a higher benefit, as it will render
        // copy operations unnecessary for the second pattern.
        patterns.insert<BufferCopyPropagation>(module.getContext(), 2);
        patterns.insert<CopyOpElimination>(module.getContext());
        (void) mlir::applyPatternsAndFoldGreedily(module, FrozenRewritePatternSet(std::move(patterns)));
      }

    };

  }
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::spn::createGPUCopyEliminationPass() {
  return std::make_unique<GPUCopyEliminationPass>();
}