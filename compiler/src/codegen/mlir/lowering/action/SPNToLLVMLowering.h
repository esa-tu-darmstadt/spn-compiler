//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_ACTIONS_SPNTOLLVMLOWERING_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_ACTIONS_SPNTOLLVMLOWERING_H

#include <driver/Actions.h>
#include <mlir/IR/Module.h>
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace spnc {

  ///
  /// Action running a series of MLIR passes on a copy of the input MLIR module.
  class SPNToLLVMLowering : public ActionSingleInput<ModuleOp, ModuleOp> {

  public:

    /// Constructor.
    /// \param _input Action providing the input MLIR module.
    /// \param _mlirContext Surrounding MLIR context.
    SPNToLLVMLowering(ActionWithOutput<ModuleOp>& _input, std::shared_ptr<MLIRContext> _mlirContext);

    ModuleOp& execute() override;

  private:

    std::shared_ptr<MLIRContext> mlirContext;

    bool cached = false;

    PassManager pm;

    std::unique_ptr<ModuleOp> module;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_ACTIONS_SPNTOLLVMLOWERING_H
