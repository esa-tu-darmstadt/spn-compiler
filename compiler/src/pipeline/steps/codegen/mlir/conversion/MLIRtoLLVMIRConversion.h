//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_MLIRTOLLVMIRCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_MLIRTOLLVMIRCONVERSION_H

#include "driver/pipeline/PipelineStep.h"
#include "mlir/IR/BuiltinOps.h"
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

namespace spnc {

  ///
  /// Action to translate MLIR LLVM dialect to actual LLVM IR.
  class MLIRtoLLVMIRConversion : public StepSingleInput<MLIRtoLLVMIRConversion, mlir::ModuleOp>,
                                 public StepWithResult<llvm::Module> {

  public:

    /// Constructor.
    /// \param _input Action providing the input MLIR module.
    /// \param context Surrounding MLIR context.
    /// \param optimizeOutput Flag indicating whether the generated LLVM IR module should be optimized
    /// after conversion.
    explicit MLIRtoLLVMIRConversion(StepWithResult<mlir::ModuleOp>& input);

    ExecutionResult executeStep(mlir::ModuleOp* mlirModule);

    llvm::Module* result() override;

    static std::string stepName;

    MLIRtoLLVMIRConversion(const MLIRtoLLVMIRConversion&) = delete;

    MLIRtoLLVMIRConversion& operator=(const MLIRtoLLVMIRConversion&) = delete;

    /// Move constructor.
    /// \param conv Move source.
    MLIRtoLLVMIRConversion(MLIRtoLLVMIRConversion&& conv) noexcept:
        StepSingleInput<MLIRtoLLVMIRConversion, mlir::ModuleOp>{conv.in},
        module{std::move(conv.module)} {}

    /// Move assignment.
    /// \param conv Move source.
    /// \return Reference to the move target.
    MLIRtoLLVMIRConversion& operator=(MLIRtoLLVMIRConversion&& conv) noexcept {
      this->in = conv.in;
      this->module = std::move(conv.module);
      return *this;
    }

    ~MLIRtoLLVMIRConversion() override {
      // We have to release the LLVM module first,
      // as the LLVMContext is deleted together with the MLIRContext
      // (because the LLVMContext's lifetime is bound to the lifetime of the LLVMDialect).
      module = nullptr;
    }

  private:

    int retrieveOptLevel();

    void optimizeLLVMIR(int optLevel);

    std::unique_ptr<llvm::Module> module;

    llvm::LLVMContext llvmCtx;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_MLIRTOLLVMIRCONVERSION_H
