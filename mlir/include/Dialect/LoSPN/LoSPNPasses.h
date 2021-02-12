//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNPASSES_H
#define SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {
    namespace low {

      std::unique_ptr<OperationPass<ModuleOp>> createLoSPNBufferizePass();

#define GEN_PASS_REGISTRATION
#include "LoSPN/LoSPNPasses.h.inc"
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNPASSES_H
