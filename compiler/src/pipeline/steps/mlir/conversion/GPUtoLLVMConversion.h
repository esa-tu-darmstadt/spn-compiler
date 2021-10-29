//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_GPUTOLLVMCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_GPUTOLLVMCONVERSION_H

#include "pipeline/PipelineStep.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "util/Logging.h"
#include <option/GlobalOptions.h>
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

namespace spnc {

  ///
  /// MLIR pass pipeline performing a series of transformations on an MLIR module
  /// to lower from GPU (and other dialects) to LLVM dialect.
  class GPUtoLLVMConversion : public StepSingleInput<GPUtoLLVMConversion, mlir::ModuleOp>,
                              public StepWithResult<mlir::ModuleOp> {

  public:

    using StepSingleInput<GPUtoLLVMConversion, mlir::ModuleOp>::StepSingleInput;

    ExecutionResult executeStep(mlir::ModuleOp* mlirModule);

    mlir::ModuleOp* result() override;

    STEP_NAME("gpu-to-llvm")

  private:

    int retrieveOptLevel();

    mlir::ModuleOp* module;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_GPUTOLLVMCONVERSION_H
