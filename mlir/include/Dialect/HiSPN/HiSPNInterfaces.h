//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_DIALECT_HISPN_HISPNINTERFACES_H
#define SPNC_MLIR_INCLUDE_DIALECT_HISPN_HISPNINTERFACES_H

#include "HiSPNEnums.h"

namespace mlir {
  namespace spn {
    namespace high {
#include "HiSPN/HiSPNInterfaces.h.inc"
}
}
}

#endif //SPNC_MLIR_INCLUDE_DIALECT_HISPN_HISPNINTERFACES_H
