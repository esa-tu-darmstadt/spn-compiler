//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PASSES_SPNPASSES_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PASSES_SPNPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {
    std::unique_ptr<Pass> createSPNSimplificationPass();
  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PASSES_SPNPASSES_H
