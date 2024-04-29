//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNPasses.h"
#include "LoSPNPassDetails.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::spn::low {
#define GEN_PASS_DEF_LOSPNCOPYREMOVAL
#include "LoSPN/LoSPNPasses.h.inc"

struct CopyRemovalPattern : public OpRewritePattern<SPNCopy> {

  using OpRewritePattern<SPNCopy>::OpRewritePattern;

  LogicalResult matchAndRewrite(SPNCopy op,
                                PatternRewriter &rewriter) const override {
    DominanceInfo domInfo(op->getParentOp());

    // Collect all users of the target memref.
    SmallVector<Operation *> tgtUsers;
    for (auto *U : op.getTarget().getUsers()) {
      if (U == op.getOperation()) {
        // Skip the copy op.
        continue;
      }
      if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(U)) {
        SmallVector<MemoryEffects::EffectInstance, 1> effects;
        memEffect.getEffectsOnValue(op.getTarget(), effects);
        for (auto e : effects) {
          if (isa<MemoryEffects::Read>(e.getEffect()) ||
              isa<MemoryEffects::Write>(e.getEffect())) {
            tgtUsers.push_back(U);
          }
        }
      }
    }

    SmallVector<Operation *> srcReads;
    SmallVector<Operation *> srcWrites;
    for (auto *U : op.getSource().getUsers()) {
      if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(U)) {
        SmallVector<MemoryEffects::EffectInstance, 1> effects;
        memEffect.getEffectsOnValue(op.getSource(), effects);
        for (auto e : effects) {
          if (isa<MemoryEffects::Read>(e.getEffect()) &&
              U != op.getOperation()) {
            srcReads.push_back(U);
          } else if (isa<MemoryEffects::Write>(e.getEffect())) {
            srcWrites.push_back(U);
          }
        }
      }
    }

    // Legality check: For the removal of the copy operation to be legal,
    // two constraints must be fulfilled:
    // 1. All users of the target memref must dominate all writes to the source
    // memref.
    //    Otherwise, they might read a wrong value (RAW) or write in the wrong
    //    order (WAW).
    // 2. All reads of the source memref must be dominated by at least one write
    // to the source memref.
    //    Otherwise, they might read values written by a write originally
    //    directed at the target memref.

    // 1. Check
    for (auto *tgtUse : tgtUsers) {
      for (auto *srcWrite : srcWrites) {
        if (!domInfo.properlyDominates(tgtUse, srcWrite)) {
          return rewriter.notifyMatchFailure(
              op, "Potential RAW/WAW hazard, abort removal");
        }
      }
    }

    // 2. Check
    for (auto *srcRead : srcReads) {
      bool dominated = false;
      for (auto *srcWrite : srcWrites) {
        if (domInfo.properlyDominates(srcWrite, srcRead)) {
          dominated = true;
          break;
        }
      }
      if (!dominated) {
        return rewriter.notifyMatchFailure(
            op, "Source read not dominated by any source write, abort removal");
      }
    }
    op.getSource().replaceAllUsesWith(op.getTarget());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct LoSPNCopyRemoval : public impl::LoSPNCopyRemovalBase<LoSPNCopyRemoval> {
  using Base::Base;

protected:
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation()->getContext());
    patterns.insert<CopyRemovalPattern>(getOperation()->getContext());
    (void)mlir::applyPatternsAndFoldGreedily(
        getOperation(), FrozenRewritePatternSet(std::move(patterns)));
  }
};

} // namespace mlir::spn::low