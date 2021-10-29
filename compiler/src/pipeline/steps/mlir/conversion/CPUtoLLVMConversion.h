//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_CPUTOLLVMCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_CPUTOLLVMCONVERSION_H

#include "pipeline/steps/mlir/MLIRPassPipeline.h"

namespace spnc {

  ///
  /// MLIR pass pipeline performing a conversion from various upstream dialects,
  /// including the Standard, MemRef, Vector and SCF dialects, to LLVM dialect.
  struct CPUtoLLVMConversion : public MLIRPassPipeline<CPUtoLLVMConversion> {
    using MLIRPassPipeline<CPUtoLLVMConversion>::MLIRPassPipeline;

    void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);

    STEP_NAME("cpu-to-llvm")

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_CPUTOLLVMCONVERSION_H
