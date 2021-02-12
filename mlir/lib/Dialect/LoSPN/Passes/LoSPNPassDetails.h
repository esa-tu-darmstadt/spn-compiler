//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_LIB_DIALECT_LOSPN_PASSES_LOSPNPASSDETAILS_H
#define SPNC_MLIR_LIB_DIALECT_LOSPN_PASSES_LOSPNPASSDETAILS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {
    namespace low {
#define GEN_PASS_CLASSES
#include "LoSPN/LoSPNPasses.h.inc"
    }
  }
}

#endif //SPNC_MLIR_LIB_DIALECT_LOSPN_PASSES_LOSPNPASSDETAILS_H
