//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_SPN_SPNPASSES_H
#define SPNC_MLIR_DIALECTS_INCLUDE_SPN_SPNPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {
    std::unique_ptr<OperationPass<ModuleOp>> createSPNOpSimplifierPass();

#define GEN_PASS_REGISTRATION
#include "SPN/SPNPasses.h.inc"
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_SPN_SPNPASSES_H
