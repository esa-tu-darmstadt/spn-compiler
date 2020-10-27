//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_PIPELINE_SPNDIALECTPIPELINE_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_PIPELINE_SPNDIALECTPIPELINE_H

#include <driver/Actions.h>
#include <mlir/IR/Module.h>
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace spnc {

  ///
  /// Action running a series of MLIR passes on the SPN dialect.
  class SPNDialectPipeline : public ActionSingleInput<mlir::ModuleOp, mlir::ModuleOp> {

  public:

    /// Constructor.
    /// \param _input Action providing the input MLIR module.
    /// \param _mlirContext Surrounding MLIR context.
    SPNDialectPipeline(ActionWithOutput<mlir::ModuleOp>& _input, std::shared_ptr<mlir::MLIRContext> _mlirContext);

    mlir::ModuleOp& execute() override;

  private:

    std::shared_ptr<mlir::MLIRContext> mlirContext;

    bool cached = false;

    mlir::PassManager pm;

    std::unique_ptr<mlir::ModuleOp> module;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_PIPELINE_SPNDIALECTPIPELINE_H
