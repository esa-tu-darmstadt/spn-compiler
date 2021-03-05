//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNINTERFACES_H
#define SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNINTERFACES_H

#include "LoSPN/LoSPNTraits.h"

namespace mlir {
  namespace spn {
    namespace low {
#include "LoSPN/LoSPNInterfaces.h.inc"
}
}
}

#endif //SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNINTERFACES_H
