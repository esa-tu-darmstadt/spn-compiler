//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#pragma once

#include "Kernel.h"
#include "mlir/IR/BuiltinOps.h"
#include "pipeline/PipelineStep.h"
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#include <poplar/Graph.hpp>

namespace spnc {

/// Step to translate MLIR Poplar dialect to a Poplar graph.
class CompilePoplarDialect
    : public StepSingleInput<CompilePoplarDialect, mlir::ModuleOp>,
      public StepWithResult<std::unique_ptr<Kernel>> {

public:
  explicit CompilePoplarDialect(StepWithResult<mlir::ModuleOp> &input);

  ExecutionResult executeStep(mlir::ModuleOp *mlirModule);

  std::unique_ptr<Kernel> *result() override { return &kernel; }

  STEP_NAME("compile-poplar-dialect")

private:
  std::unique_ptr<Kernel> kernel;
};

} // namespace spnc
