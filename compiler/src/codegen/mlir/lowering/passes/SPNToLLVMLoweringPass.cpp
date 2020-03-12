//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <codegen/mlir/dialects/spn/SPNDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <codegen/mlir/lowering/patterns/LowerSPNtoLLVMPatterns.h>
#include "SPNLoweringPasses.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"

using namespace mlir;
using namespace mlir::spn;

namespace {

  struct SPNtoLLVMLoweringPass : public ModulePass<SPNtoLLVMLoweringPass> {

    void runOnModule() override {

      ConversionTarget target(getContext());

      // Only operations from the LLVM dialect and modules will be legal after running this pass.
      target.addLegalDialect<LLVM::LLVMDialect>();
      target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

      // Mark all operations from the SPN and Standard dialect as illegal.
      target.addIllegalDialect<SPNDialect>();
      target.addIllegalDialect<StandardOpsDialect>();

      LLVMTypeConverter typeConverter(&getContext());

      // Create and populate list of pattern used for conversion.
      OwningRewritePatternList patterns;
      populateStdToLLVMConversionPatterns(typeConverter, patterns);

      patterns.insert<HistogramValueLowering>(&getContext(), typeConverter);

      auto module = getModule();
      // Apply the full conversion to the module.
      if (failed(applyFullConversion(module, target, patterns, &typeConverter))) {
        signalPassFailure();
      }

    }

  };

}

std::unique_ptr<Pass> mlir::spn::createSPNtoLLVMLoweringPass() {
  return std::make_unique<SPNtoLLVMLoweringPass>();
}
