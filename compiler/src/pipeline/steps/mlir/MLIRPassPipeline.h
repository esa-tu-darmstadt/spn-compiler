//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRPASSPIPELINE_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRPASSPIPELINE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "option/Options.h"
#include "pipeline/PipelineStep.h"
#include "util/Logging.h"

namespace spnc {

///
/// Base for pass pipelines operating on MLIR modules.
/// \tparam PassPipeline CRTP template parameter, inheriting classes need to
/// provide a method 'void initializePassPipeline(mlir::PassManager*,
/// mlir::MLIRContext*)'.
template <typename PassPipeline>
struct MLIRPassPipeline : public StepSingleInput<PassPipeline, mlir::ModuleOp>,
                          public StepWithResult<mlir::ModuleOp> {

public:
  using StepSingleInput<PassPipeline, mlir::ModuleOp>::StepSingleInput;

  ExecutionResult executeStep(mlir::ModuleOp *module) {
    mlir::MLIRContext *ctx =
        this->getContext()->template get<mlir::MLIRContext>();
    mlir::PassManager pm{ctx};
    static_cast<PassPipeline *>(this)->initializePassPipeline(&pm, ctx);
    // Enable IR printing if requested via CLI
    if (spnc::option::dumpIR) {
      pm.enableIRPrinting(
          /* Print before every pass*/ [](mlir::Pass *,
                                          mlir::Operation *) { return false; },
          /* Print after every pass*/
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /* Print module scope*/ true,
          /* Print only after change*/ false);
    }
    // Invoke the pre-processing defined by the CRTP heirs of this class
    static_cast<PassPipeline *>(this)->preProcess(module);
    auto result = pm.run(*module);
    if (failed(result)) {
      return spnc::failure("Running the MLIR pass pipeline failed");
    }
    auto verificationResult = module->verify();
    if (failed(verificationResult)) {
      return spnc::failure("Transformed module failed verification");
    }
    // Invoke the post-processing defined by the CRTP heirs of this class
    static_cast<PassPipeline *>(this)->postProcess(module);

    theModule = module;

    return spnc::success();
  }

  mlir::ModuleOp *result() override { return theModule; }

  virtual void postProcess(mlir::ModuleOp *transformedModule){};

  virtual void preProcess(mlir::ModuleOp *inputModule){};

private:
  mlir::ModuleOp *theModule = nullptr;
};

} // namespace spnc

#endif // SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRPASSPIPELINE_H
