//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_GPUTOLLVMCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_GPUTOLLVMCONVERSION_H

#include <driver/Actions.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "util/Logging.h"
#include <driver/GlobalOptions.h>
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

namespace spnc {

  ///
  /// Action performing a series of transformations on an MLIR module
  /// to lower from GPU (and other dialects) to LLVM dialect.
  class GPUtoLLVMConversion : public ActionSingleInput<mlir::ModuleOp, mlir::ModuleOp> {

  public:

    GPUtoLLVMConversion(ActionWithOutput<mlir::ModuleOp>& input,
                        std::shared_ptr<mlir::MLIRContext> ctx);

    mlir::ModuleOp& execute() override;

  private:

    static mlir::OwnedBlob compilePtxToCubin(std::string ptx, mlir::Location loc,
                                             llvm::StringRef name);

    struct GPUOptimizationOptions {
      std::string computeCapability;
      std::string features;
      bool flushDenormalsToZero;
      bool printIRAfter;
    };

    static std::unique_ptr<llvm::Module> translateAndLinkGPUModule(llvm::ArrayRef<llvm::StringRef> kernelFuncs,
                                                                   const GPUOptimizationOptions& options,
                                                                   mlir::Operation* gpuModule,
                                                                   llvm::LLVMContext& llvmContext,
                                                                   llvm::StringRef name = "LLVMDialectModule"
    );

    static void linkWithLibdevice(llvm::Module* module, llvm::ArrayRef<llvm::StringRef> kernelFuncs);

    static void optimizeNVVMIR(llvm::Module* module, const GPUOptimizationOptions& options);

    static std::string getGPUArchitecture();

    bool cached = false;

    std::shared_ptr<mlir::MLIRContext> mlirContext;

    std::unique_ptr<mlir::ModuleOp> module;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_GPUTOLLVMCONVERSION_H
