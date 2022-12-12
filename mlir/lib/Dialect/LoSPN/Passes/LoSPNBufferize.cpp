//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

//#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
//#include "mlir/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "LoSPNPassDetails.h"
#include "LoSPN/LoSPNPasses.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "../Bufferize/LoSPNBufferizationPatterns.h"

using namespace mlir;
using namespace mlir::spn::low;

namespace {

  struct LoSPNBufferize : public LoSPNBufferizeBase<LoSPNBufferize> {
  protected:
    void runOnOperation() override {
      ConversionTarget target(getContext());

      target.addLegalDialect<LoSPNDialect>();
      target.addLegalDialect<arith::ArithDialect>();
      target.addLegalDialect<mlir::memref::MemRefDialect>();
      target.addLegalDialect<mlir::bufferization::BufferizationDialect>();
      target.addLegalOp<ModuleOp, mlir::func::FuncOp>();

      target.addIllegalOp<SPNBatchExtract, SPNBatchCollect>();
      bufferization::BufferizeTypeConverter typeConverter;
      target.addDynamicallyLegalOp<SPNTask>([&](SPNTask op) {
        if (!op.getResults().empty()) {
          return false;
        }
        for (auto in : op.getInputs()) {
          if (!typeConverter.isLegal(in.getType())) {
            return false;
          }
        }
        return true;
      });
      target.addDynamicallyLegalOp<SPNKernel>([&](SPNKernel op) {
        return typeConverter.isSignatureLegal(op.getFunctionType());
      });
      target.addDynamicallyLegalOp<SPNReturn>([&](SPNReturn op) {
        return std::all_of(op->result_begin(), op->result_end(), [&](OpResult res) {
          return typeConverter.isLegal(res.getType()) && !res.getType().isa<MemRefType>();
        });
      });

      RewritePatternSet patterns(&getContext());
      mlir::spn::low::populateLoSPNBufferizationPatterns(patterns, &getContext(), typeConverter);

      auto op = getOperation();
      FrozenRewritePatternSet frozenPatterns(std::move(patterns));
      if (failed(applyPartialConversion(op, target, frozenPatterns))) {
        signalPassFailure();
      }
    }
  };

}

std::unique_ptr<OperationPass<ModuleOp>> mlir::spn::low::createLoSPNBufferizePass() {
  return std::make_unique<LoSPNBufferize>();
}