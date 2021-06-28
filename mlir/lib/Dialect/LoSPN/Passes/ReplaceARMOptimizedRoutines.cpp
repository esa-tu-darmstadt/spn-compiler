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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "LoSPNPassDetails.h"
#include "LoSPN/LoSPNPasses.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
  namespace spn {
    namespace low {

      template<typename UnOp>
      class ReplaceUnary : public OpRewritePattern<UnOp> {

        using OpRewritePattern<UnOp>::OpRewritePattern;

      public:

        LogicalResult matchAndRewrite(UnOp op, PatternRewriter& rewriter) const override {
          // Perform checks, the replacement is only performed for 1D vectors.
          if (op.operand().getType() != op.getResult().getType()) {
            return rewriter.notifyMatchFailure(op, "Type conversion is not supported");
          }
          VectorType vecType = op.result().getType().template dyn_cast<VectorType>();
          if (!vecType || vecType.getShape().size() != 1) {
            return rewriter.notifyMatchFailure(op, "Replacement is only supported for 1D vectors");
          }
          // Check if the concrete pattern defines a replacement for the given vector shape and element type.
          llvm::Optional<StringRef> funcName = getReplacementFunction(vecType);
          if (!funcName) {
            return rewriter.notifyMatchFailure(op, "No substitution defined for vector type and/or shape");
          }
          // Check if the replacement function is already present in the module (and it's symbol table).
          // If not, create a new external function.
          auto module = op->template getParentOfType<ModuleOp>();
          FuncOp replaceFunc = module.template lookupSymbol<mlir::FuncOp>(funcName.getValue());
          if (!replaceFunc) {
            auto funcType = rewriter.getFunctionType(op.operand().getType(), op.result().getType());
            auto restore = rewriter.saveInsertionPoint();
            rewriter.setInsertionPointToEnd(module.getBody(0));
            // External functions must not have public visibility, so it's marked private here.
            replaceFunc = rewriter.create<mlir::FuncOp>(op->getLoc(), funcName.getValue(), funcType,
                                                        rewriter.getStringAttr("private"));
            rewriter.restoreInsertionPoint(restore);
          }
          rewriter.replaceOpWithNewOp<mlir::CallOp>(op, replaceFunc, op.operand());
          return mlir::success();
        }

      protected:

        virtual llvm::Optional<StringRef> getReplacementFunction(VectorType vecTy) const = 0;

      private:

        llvm::DenseMap<VectorType*, llvm::Optional<FuncOp>> cache;

      };

      class ReplaceExp : public ReplaceUnary<math::ExpOp> {

        using ReplaceUnary<math::ExpOp>::ReplaceUnary;

      protected:
        Optional<StringRef> getReplacementFunction(VectorType vecTy) const override {
          auto elemTy = vecTy.getElementType();
          auto dim = vecTy.getShape().front();
          if (dim == 2 && elemTy.isF64()) {
            return StringRef("__v_exp");
          }
          if (dim == 4 && elemTy.isF32()) {
            return StringRef("__v_expf");
          }
          return llvm::None;
        }
      };

      class ReplaceLog : public ReplaceUnary<math::LogOp> {

        using ReplaceUnary<math::LogOp>::ReplaceUnary;

      protected:
        Optional<StringRef> getReplacementFunction(VectorType vecTy) const override {
          auto elemTy = vecTy.getElementType();
          auto dim = vecTy.getShape().front();
          if (dim == 2 && elemTy.isF64()) {
            return StringRef("__v_log");
          }
          if (dim == 4 && elemTy.isF32()) {
            return StringRef("__v_logf");
          }
          return llvm::None;
        }
      };

      struct ReplaceARMOptimizedRoutines : public ReplaceARMOptimizedRoutinesBase<ReplaceARMOptimizedRoutines> {

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

    }
  }
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::spn::low::createReplaceARMOptimizedRoutinesPass() {
  return std::make_unique<ReplaceARMOptimizedRoutines>();
}