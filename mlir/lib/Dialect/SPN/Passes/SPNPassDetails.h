//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_LIB_SPN_PASSES_SPNDIALECTPASSES_H
#define SPNC_MLIR_DIALECTS_LIB_SPN_PASSES_SPNDIALECTPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {
#define GEN_PASS_CLASSES
#include "SPN/SPNPasses.h.inc"
  }
}

#endif //SPNC_MLIR_DIALECTS_LIB_SPN_PASSES_SPNDIALECTPASSES_H
