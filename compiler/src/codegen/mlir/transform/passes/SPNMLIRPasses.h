//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PASSES_SPNPASSES_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PASSES_SPNPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {
    /// Instantiate the simplification pass.
    /// \return Owning pointer to the created MLIR pass.
    std::unique_ptr<Pass> createSPNSimplificationPass();
    /// Instantiate the canonicalization pass.
    /// \return Owning pointer to the created MLIR pass.
    std::unique_ptr<Pass> createSPNCanonicalizationPass();
  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PASSES_SPNPASSES_H
