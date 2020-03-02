//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PASSES_SPNLOWERINGPASSES_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PASSES_SPNLOWERINGPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {
    std::unique_ptr<Pass> createSPNtoStandardLoweringPass();
    std::unique_ptr<Pass> createSPNtoLLVMLoweringPass();
  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PASSES_SPNLOWERINGPASSES_H
