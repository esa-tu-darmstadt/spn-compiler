//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_GPUTOLLVMCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_GPUTOLLVMCONVERSION_H

#include <driver/Actions.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "util/Logging.h"
#include <driver/GlobalOptions.h>
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

namespace spnc {

  ///
  /// Action performing a series of transformations on an MLIR module
  /// to lower from GPU (and other dialects) to LLVM dialect.
  class GPUtoLLVMConversion : public ActionSingleInput<mlir::ModuleOp, mlir::ModuleOp> {

  public:

    GPUtoLLVMConversion(ActionWithOutput<mlir::ModuleOp>& input,
                        std::shared_ptr<mlir::MLIRContext> ctx, unsigned optLevel);

    mlir::ModuleOp& execute() override;

  private:

    bool cached = false;

    std::shared_ptr<mlir::MLIRContext> mlirContext;

    unsigned irOptLevel;

    std::unique_ptr<mlir::ModuleOp> module;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_GPUTOLLVMCONVERSION_H
