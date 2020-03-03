//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_DRIVER_ACTION_MLIRTOLLVMCONVERSION_H
#define SPNC_COMPILER_SRC_DRIVER_ACTION_MLIRTOLLVMCONVERSION_H

#include <driver/Actions.h>
#include <mlir/IR/Module.h>
#include <llvm/IR/Module.h>

namespace spnc {

  class MLIRtoLLVMConversion : public ActionSingleInput<mlir::ModuleOp, llvm::Module> {

  public:

    explicit MLIRtoLLVMConversion(ActionWithOutput<mlir::ModuleOp>& _input,
                                  std::shared_ptr<mlir::MLIRContext> context, bool optimizeOutput = true);

    llvm::Module& execute() override;

    MLIRtoLLVMConversion(const MLIRtoLLVMConversion&) = delete;

    MLIRtoLLVMConversion& operator=(const MLIRtoLLVMConversion&) = delete;

    MLIRtoLLVMConversion(MLIRtoLLVMConversion&& conv) noexcept :
        ActionSingleInput<mlir::ModuleOp, llvm::Module>{conv.input},
        module{std::move(conv.module)}, ctx{std::move(conv.ctx)},
        cached{conv.cached}, optimize{conv.optimize} {
      conv.cached = false;
    }

    MLIRtoLLVMConversion& operator=(MLIRtoLLVMConversion&& conv) noexcept {
      this->input = conv.input;
      this->module = std::move(conv.module);
      this->ctx = std::move(conv.ctx);
      this->cached = conv.cached;
      conv.cached = false;
      this->optimize = conv.optimize;
      return *this;
    }

    ~MLIRtoLLVMConversion() override {
      // We have to release the LLVM module first,
      // as the LLVMContext is deleted together with the MLIRContext
      // (because the LLVMContext's lifetime is bound to the lifetime of the LLVMDialect).
      module = nullptr;
      ctx.reset();
    }

  private:

    std::unique_ptr<llvm::Module> module;

    bool cached = false;

    bool optimize;

    std::shared_ptr<mlir::MLIRContext> ctx;

  };

}

#endif //SPNC_COMPILER_SRC_DRIVER_ACTION_MLIRTOLLVMCONVERSION_H
