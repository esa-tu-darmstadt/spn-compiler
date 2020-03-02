//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/StandardOps/Ops.h>
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

      target.addLegalDialect<LLVM::LLVMDialect>();
      target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

      target.addIllegalDialect<SPNDialect>();
      target.addIllegalDialect<StandardOpsDialect>();

      LLVMTypeConverter typeConverter(&getContext());

      OwningRewritePatternList patterns;
      populateStdToLLVMConversionPatterns(typeConverter, patterns);

      patterns.insert<HistogramValueLowering>(&getContext(), typeConverter);

      auto module = getModule();
      if (failed(applyFullConversion(module, target, patterns, &typeConverter))) {
        signalPassFailure();
      }

    }

  };

}

std::unique_ptr<Pass> mlir::spn::createSPNtoLLVMLoweringPass() {
  return std::make_unique<SPNtoLLVMLoweringPass>();
}
