//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Bufferize.h"
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
      target.addLegalDialect<StandardOpsDialect>();
      target.addLegalOp<ModuleOp, ModuleTerminatorOp, FuncOp>();

      target.addIllegalOp<SPNBatchExtract, SPNBatchCollect>();
      BufferizeTypeConverter typeConverter;
      target.addDynamicallyLegalOp<SPNTask>([&](SPNTask op) {
        if (!op.results().empty()) {
          return false;
        }
        for (auto in : op.inputs()) {
          if (!typeConverter.isLegal(in.getType())) {
            return false;
          }
        }
        return true;
      });
      target.addDynamicallyLegalOp<SPNKernel>([&](SPNKernel op) {
        return typeConverter.isSignatureLegal(op.getType());
      });
      target.addDynamicallyLegalOp<SPNReturn>([&](SPNReturn op) {
        return std::all_of(op->result_begin(), op->result_end(), [&](OpResult res) {
          return typeConverter.isLegal(res.getType()) && !res.getType().isa<MemRefType>();
        });
      });

      OwningRewritePatternList patterns;
      mlir::spn::low::populateLoSPNBufferizationPatterns(patterns, &getContext(), typeConverter);

      auto op = getOperation();
      FrozenRewritePatternList frozenPatterns(std::move(patterns));
      if (failed(applyPartialConversion(op, target, frozenPatterns))) {
        signalPassFailure();
      }
    }
  };

}

std::unique_ptr<OperationPass<ModuleOp>> mlir::spn::low::createLoSPNBufferizePass() {
  return std::make_unique<LoSPNBufferize>();
}