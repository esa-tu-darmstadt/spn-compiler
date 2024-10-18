//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "../Bufferize/LoSPNBufferizationPatterns.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNPasses.h"
#include "LoSPNPassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::spn::low;

namespace mlir::spn::low {

#define GEN_PASS_DEF_LOSPNBUFFERIZE
#include "LoSPN/LoSPNPasses.h.inc"

struct LoSPNBufferize : public impl::LoSPNBufferizeBase<LoSPNBufferize> {
protected:
  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<LoSPNDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::bufferization::BufferizationDialect>(); // CHECK
                                                                         // ME
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();

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
      FunctionOpInterface functionInterface = cast<FunctionOpInterface>(op.getOperation());
      auto funcType = functionInterface.getFunctionType();
      // llvm::outs() << "Checking legality of SPNKernel: " << funcType << "\n";
      // llvm::outs() << "Is its signature legal? " <<
      // typeConverter.isSignatureLegal(funcType.cast<FunctionType>()) << "\n";
      assert(funcType.isa<FunctionType>() && "SPNKernel must have a FunctionType");
      return typeConverter.isSignatureLegal(funcType.cast<FunctionType>());
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

} // namespace mlir::spn::low