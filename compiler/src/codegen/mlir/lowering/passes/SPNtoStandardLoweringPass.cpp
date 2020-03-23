//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
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

      // All operations from the Standard dialect and modules will be allowed after this pass.
      target.addLegalDialect<StandardOpsDialect>();
      target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

      // Instantiate the type converter for function signature rewrites.
      SPNTypeConverter typeConverter;

      // Functions are dynamically legal, i.e. they are only legal if
      // their signature has been converted to use MemRef instead of Tensor.
      target.addDynamicallyLegalOp<FuncOp>([&typeConverter](FuncOp op) {
        return typeConverter.isSignatureLegal(op.getType());
      });

      // Mark all operations from the SPN dialect except the HistogramValueOp as illegal.
      // The HistogramValueOp will be converted directly into LLVM dialect later on.
      target.addIllegalDialect<SPNDialect>();
      target.addLegalOp<HistogramValueOp>();

      // Create and populate the list of patterns used for conversion.
      OwningRewritePatternList patterns;
      mlir::spn::populateSPNtoStandardConversionPatterns(patterns, &getContext(), typeConverter);

      auto module = getModule();
      // Apply the conversion.
      if (failed(applyFullConversion(module, target, patterns, &typeConverter))) {
        signalPassFailure();
      }
    }

  };

}

std::unique_ptr<Pass> mlir::spn::createSPNtoStandardLoweringPass() {
  return std::make_unique<SPNtoStandardLoweringPass>();
}