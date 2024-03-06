//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPN/LoSPNPasses.h"
#include "LoSPNPassDetails.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <optional>

namespace mlir {
namespace spn {
namespace low {

#define GEN_PASS_DECL_REPLACEARMOPTIMIZEDROUTINES
#define GEN_PASS_DEF_REPLACEARMOPTIMIZEDROUTINES
#include "LoSPN/LoSPNPasses.h.inc"

template <typename UnOp>
class ReplaceUnary : public OpRewritePattern<UnOp> {

  using OpRewritePattern<UnOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(UnOp op,
                                PatternRewriter &rewriter) const override {
    // Perform checks, the replacement is only performed for 1D vectors.
    if (op.getOperand().getType() != op.getResult().getType()) {
      return rewriter.notifyMatchFailure(op,
                                         "Type conversion is not supported");
    }
    VectorType vecType =
        op.getResult().getType().template dyn_cast<VectorType>();
    if (!vecType || vecType.getShape().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Replacement is only supported for 1D vectors");
    }
    // Check if the concrete pattern defines a replacement for the given vector
    // shape and element type.
    std::optional<StringRef> funcName = getReplacementFunction(vecType);
    if (!funcName) {
      return rewriter.notifyMatchFailure(
          op, "No substitution defined for vector type and/or shape");
    }
    // Check if the replacement function is already present in the module (and
    // it's symbol table). If not, create a new external function.
    auto module = op->template getParentOfType<ModuleOp>();
    func::FuncOp replaceFunc =
        module.template lookupSymbol<func::FuncOp>(funcName.value());
    if (!replaceFunc) {
      auto funcType = rewriter.getFunctionType(op.getOperand().getType(),
                                               op.getResult().getType());
      auto restore = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToEnd(module.getBody(0));
      // External functions must not have public visibility, so it's marked
      // private here.
      replaceFunc = rewriter.create<func::FuncOp>(op->getLoc(), funcName.value(), funcType/*,
                                                        rewriter.getStringAttr("private")*/);
      rewriter.restoreInsertionPoint(restore);
    }
    rewriter.replaceOpWithNewOp<func::CallOp>(op, replaceFunc, op.getOperand());
    return mlir::success();
  }

protected:
  virtual std::optional<StringRef>
  getReplacementFunction(VectorType vecTy) const = 0;

private:
  llvm::DenseMap<VectorType *, std::optional<func::FuncOp>> cache;
};

class ReplaceExp : public ReplaceUnary<math::ExpOp> {

  using ReplaceUnary<math::ExpOp>::ReplaceUnary;

protected:
  std::optional<StringRef>
  getReplacementFunction(VectorType vecTy) const override {
    auto elemTy = vecTy.getElementType();
    auto dim = vecTy.getShape().front();
    if (dim == 2 && elemTy.isF64()) {
      return StringRef("__v_exp");
    }
    if (dim == 4 && elemTy.isF32()) {
      return StringRef("__v_expf");
    }
    return std::nullopt;
  }
};

class ReplaceLog : public ReplaceUnary<math::LogOp> {

  using ReplaceUnary<math::LogOp>::ReplaceUnary;

protected:
  std::optional<StringRef>
  getReplacementFunction(VectorType vecTy) const override {
    auto elemTy = vecTy.getElementType();
    auto dim = vecTy.getShape().front();
    if (dim == 2 && elemTy.isF64()) {
      return StringRef("__v_log");
    }
    if (dim == 4 && elemTy.isF32()) {
      return StringRef("__v_logf");
    }
    return std::nullopt;
  }
};

struct ReplaceARMOptimizedRoutines
    : public impl::ReplaceARMOptimizedRoutinesBase<
          ReplaceARMOptimizedRoutines> {

protected:
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation()->getContext());
    patterns.insert<ReplaceExp, ReplaceLog>(getOperation()->getContext());
    mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace low
} // namespace spn
} // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::spn::low::createReplaceARMOptimizedRoutinesPass() {
  return std::make_unique<ReplaceARMOptimizedRoutines>();
}