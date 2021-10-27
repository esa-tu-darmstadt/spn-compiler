//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_HISPNTOLOSPNCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_HISPNTOLOSPNCONVERSION_H

#include "pipeline/steps/codegen/mlir/MLIRPassPipeline.h"

namespace spnc {

  struct HiSPNtoLoSPNConversion : public MLIRPassPipeline<HiSPNtoLoSPNConversion> {

    using MLIRPassPipeline<HiSPNtoLoSPNConversion>::MLIRPassPipeline;

    void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);

    static std::string stepName;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_HISPNTOLOSPNCONVERSION_H
