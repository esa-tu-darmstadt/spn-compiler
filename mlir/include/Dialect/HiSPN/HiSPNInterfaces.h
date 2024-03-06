//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_DIALECT_HISPN_HISPNINTERFACES_H
#define SPNC_MLIR_INCLUDE_DIALECT_HISPN_HISPNINTERFACES_H

#include "HiSPNEnums.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
  namespace spn {
    namespace high {
#include "HiSPN/HiSPNInterfaces.h.inc"
}
}
}

#endif //SPNC_MLIR_INCLUDE_DIALECT_HISPN_HISPNINTERFACES_H
