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
#include <driver/GlobalOptions.h>

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
        // Enable IR printing if requested via CLI
        if (spnc::option::dumpIR.get(*this->config)) {
          pm.enableIRPrinting(/* Print before every pass*/ [](mlir::Pass*, mlir::Operation*) { return false; },
              /* Print after every pass*/ [](mlir::Pass*, mlir::Operation*) { return true; },
              /* Print module scope*/ true,
              /* Print only after change*/ false);
        }
        auto inputModule = input.execute();
        // Clone the module to keep the original module available
        // for actions using the same input module.
        module = std::make_unique<mlir::ModuleOp>(inputModule.clone());
        // Invoke the pre-processing defined by the CRTP heirs of this class
        static_cast<PassPipeline*>(this)->preProcess(module.get());
        auto result = pm.run(*module);
        if (failed(result)) {
          SPNC_FATAL_ERROR("Running the MLIR pass pipeline failed!");
        }
        auto verificationResult = module->verify();
        if (failed(verificationResult)) {
          SPNC_FATAL_ERROR("Transformed module failed verification");
        }
        // Invoke the post-processing defined by the CRTP heirs of this class
        static_cast<PassPipeline*>(this)->postProcess(module.get());
        cached = true;
      }
      return *module;
    }

    virtual void postProcess(mlir::ModuleOp* transformedModule) {};

    virtual void preProcess(mlir::ModuleOp* inputModule) {};

  private:

    std::shared_ptr<mlir::MLIRContext> mlirContext;

    std::shared_ptr<mlir::ScopedDiagnosticHandler> diagnostics;

    bool cached = false;

    mlir::PassManager pm;

    std::unique_ptr<mlir::ModuleOp> module;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRPASSPIPELINE_H
