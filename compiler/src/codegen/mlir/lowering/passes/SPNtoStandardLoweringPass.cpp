//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/StandardOps/Ops.h>
#include <codegen/mlir/dialects/spn/SPNDialect.h>
#include <codegen/mlir/lowering/types/SPNTypeConverter.h>
#include "SPNLoweringPasses.h"
#include <codegen/mlir/lowering/patterns/LowerSPNtoStandardPatterns.h>
#include <algorithm>

using namespace mlir;
using namespace mlir::spn;

namespace {

  struct SPNtoStandardLoweringPass : public ModulePass<SPNtoStandardLoweringPass> {

    void runOnModule() override {

      ConversionTarget target(getContext());

      target.addLegalDialect<StandardOpsDialect>();

      target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
      target.addDynamicallyLegalOp<FuncOp>([](FuncOp op) {
        auto fnType = op.getType();
        return std::none_of(fnType.getInputs().begin(), fnType.getInputs().end(), [](Type t) {
          return t.isa<TensorType>();
        });
      });

      target.addIllegalDialect<SPNDialect>();
      target.addLegalOp<HistogramValueOp>();

      SPNTypeConverter typeConverter;

      OwningRewritePatternList patterns;
      mlir::spn::populateSPNtoStandardConversionPatterns(patterns, &getContext(), typeConverter);

      auto module = getModule();
      if (failed(applyFullConversion(module, target, patterns, &typeConverter))) {
        signalPassFailure();
      }
    }

  };

}

std::unique_ptr<Pass> mlir::spn::createSPNtoStandardLoweringPass() {
  return std::make_unique<SPNtoStandardLoweringPass>();
}