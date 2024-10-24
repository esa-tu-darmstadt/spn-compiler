//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "CPUtoPoplar/CPUtoPoplarConversionPasses.h"
#include "LoSPN/LoSPNDialect.h"
#include "mlir-ipu/Dialect/Poplar/IR/Poplar.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

/// Matches func::FuncOp's that are tasks and converts them to
/// ipu::poplar::CodeletOp's
/// Tasks are functions that have only memref arguments, do not return any value
/// and are named task*
struct TaskConversionPattern : public OpConversionPattern<func::FuncOp> {

  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult match(func::FuncOp op) const override {
    // Make sure that all arguments are memrefs
    for (BlockArgument &arg : op.getArguments()) {
      if (!arg.getType().isa<MemRefType>()) {
        return failure();
      }
    }

    // Make sure that is is a task, and not a kernel
    // FIXME: We obviously need a better way to identify tasks
    if (!op.getSymName().starts_with("task"))
      return failure();

    return success();
  }

  void rewrite(func::FuncOp funcOp, func::FuncOp::Adaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    auto codelet = rewriter.create<ipu::poplar::CodeletOp>(
        funcOp.getLoc(), funcOp.getName(), funcOp.getFunctionType());

    rewriter.moveBlockBefore(&funcOp.front(),
                             &codelet.getFunctionBody().front());
    rewriter.replaceOp(funcOp, codelet);
  }
};

/// Matches func::ReturnOp's inside a CodeletOp's and converts them to
/// ipu::poplar::ReturnOp's
struct ReturnConversionPattern : public OpConversionPattern<func::ReturnOp> {

  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult match(func::ReturnOp op) const override {
    // Match a func::ReturnOp inside a CodeletOp
    if (op->getParentOfType<ipu::poplar::CodeletOp>())
      return success();

    return failure();
  }

  void rewrite(func::ReturnOp funcRetOp, func::ReturnOp::Adaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    auto codeletRetOp = rewriter.create<ipu::poplar::ReturnOp>(
        funcRetOp->getLoc(), adaptor.getOperands());

    rewriter.replaceOp(funcRetOp, codeletRetOp);
  }
};

/// Matches func::FuncOp's that are kernels and converts them to
/// ipu::poplar::GraphOp's
/// Kernels are functions that have only memref arguments, do not return any
/// value and are named spn_kernel
struct KernelConversionPattern : public OpConversionPattern<func::FuncOp> {

  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult match(func::FuncOp op) const override {
    // Make sure that all arguments are memrefs
    for (BlockArgument &arg : op.getArguments()) {
      if (!arg.getType().isa<MemRefType>()) {
        return failure();
      }
    }

    // Make sure that is is a task, and not a kernel
    // FIXME: We obviously need a better way to identify kernels
    if (op.getSymName() != "spn_kernel")
      return failure();

    return success();
  }

  void rewrite(func::FuncOp funcOp, func::FuncOp::Adaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    auto graph = rewriter.create<ipu::poplar::GraphOp>(
        funcOp.getLoc(), ipu::poplar::GraphType::get(rewriter.getContext()));

    rewriter.createBlock(&graph.getBodyRegion());
    rewriter.setInsertionPointToStart(&graph.getBodyRegion().front());
  }
};
} // namespace

namespace mlir {
namespace spn {
#define GEN_PASS_DEF_CPUTOPOPLARCONVERSIONPASS
#include "CPUtoPoplar/CPUtoPoplarConversionPasses.h.inc"

struct CPUtoPoplarConversionPass
    : public impl::CPUtoPoplarConversionPassBase<CPUtoPoplarConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalOp<func::FuncOp>();
    target.addIllegalOp<func::ReturnOp>();
    // target.addLegalOp<ModuleOp>();
    target.addLegalDialect<ipu::poplar::PoplarDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<TaskConversionPattern>(&getContext());
    patterns.add<ReturnConversionPattern>(&getContext());
    patterns.add<KernelConversionPattern>(&getContext());

    ModuleOp op = getOperation();
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(op, target, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace spn
} // namespace mlir