//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRPASSPIPELINE_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRPASSPIPELINE_H

#include <driver/Actions.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "util/Logging.h"

namespace spnc {

  ///
  /// Base for pass pipelines operating on MLIR modules.
  /// \tparam PassPipeline CRTP template parameter, inheriting classes need to
  /// provide a method 'void initializePassPipeline(mlir::PassManager*, mlir::MLIRContext*)'.
  template<typename PassPipeline>
  struct MLIRPipelineBase : ActionSingleInput<mlir::ModuleOp, mlir::ModuleOp> {

  public:
    MLIRPipelineBase(ActionWithOutput <mlir::ModuleOp>& _input,
                     std::shared_ptr<mlir::MLIRContext> ctx, std::shared_ptr<mlir::ScopedDiagnosticHandler> handler)
        : ActionSingleInput<mlir::ModuleOp, mlir::ModuleOp>{_input}, mlirContext{std::move(ctx)},
          diagnostics{std::move(handler)}, pm{mlirContext.get()} {}

    mlir::ModuleOp& execute() override {
      if (!cached) {
        static_cast<PassPipeline*>(this)->initializePassPipeline(&pm, mlirContext.get());
        auto inputModule = input.execute();
        // Clone the module to keep the original module available
        // for actions using the same input module.
        module = std::make_unique<mlir::ModuleOp>(inputModule.clone());
        auto result = pm.run(*module);
        if (failed(result)) {
          SPNC_FATAL_ERROR("Running the MLIR pass pipeline failed!");
        }
        cached = true;
      }
      module->dump();
      return *module;
    }

  private:

    std::shared_ptr<mlir::MLIRContext> mlirContext;

    std::shared_ptr<mlir::ScopedDiagnosticHandler> diagnostics;

    bool cached = false;

    mlir::PassManager pm;

    std::unique_ptr<mlir::ModuleOp> module;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRPASSPIPELINE_H
