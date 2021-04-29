//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_MLIRTOLLVMIRCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_MLIRTOLLVMIRCONVERSION_H

#include <driver/Actions.h>
#include "mlir/IR/BuiltinOps.h"
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

namespace spnc {

  ///
  /// Action to translate MLIR LLVM dialect to actual LLVM IR.
  class MLIRtoLLVMIRConversion : public ActionSingleInput<mlir::ModuleOp, llvm::Module> {

  public:

    /// Constructor.
    /// \param _input Action providing the input MLIR module.
    /// \param context Surrounding MLIR context.
    /// \param optimizeOutput Flag indicating whether the generated LLVM IR module should be optimized
    /// after conversion.
    explicit MLIRtoLLVMIRConversion(ActionWithOutput<mlir::ModuleOp>& _input,
                                    std::shared_ptr<mlir::MLIRContext> context,
                                    std::shared_ptr<llvm::TargetMachine> targetMachine,
                                    bool optimizeOutput = true);

    llvm::Module& execute() override;

    MLIRtoLLVMIRConversion(const MLIRtoLLVMIRConversion&) = delete;

    MLIRtoLLVMIRConversion& operator=(const MLIRtoLLVMIRConversion&) = delete;

    /// Move constructor.
    /// \param conv Move source.
    MLIRtoLLVMIRConversion(MLIRtoLLVMIRConversion&& conv) noexcept:
        ActionSingleInput<mlir::ModuleOp, llvm::Module>{conv.input},
        module{std::move(conv.module)}, cached{conv.cached}, 
        optimize{conv.optimize}, ctx{std::move(conv.ctx)} {
      conv.cached = false;
    }

    /// Move assignment.
    /// \param conv Move source.
    /// \return Reference to the move target.
    MLIRtoLLVMIRConversion& operator=(MLIRtoLLVMIRConversion&& conv) noexcept {
      this->input = conv.input;
      this->module = std::move(conv.module);
      this->ctx = std::move(conv.ctx);
      this->cached = conv.cached;
      conv.cached = false;
      this->optimize = conv.optimize;
      return *this;
    }

    ~MLIRtoLLVMIRConversion() override {
      // We have to release the LLVM module first,
      // as the LLVMContext is deleted together with the MLIRContext
      // (because the LLVMContext's lifetime is bound to the lifetime of the LLVMDialect).
      module = nullptr;
      ctx.reset();
    }

  private:

    std::unique_ptr<llvm::Module> module;

    bool cached;

    bool optimize;

    std::shared_ptr<mlir::MLIRContext> ctx;

    std::shared_ptr<llvm::TargetMachine> machine;

    llvm::LLVMContext llvmCtx;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_MLIRTOLLVMIRCONVERSION_H
