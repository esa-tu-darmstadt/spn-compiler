//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_LIB_DIALECT_LOSPN_PASSES_LOSPNPASSDETAILS_H
#define SPNC_MLIR_LIB_DIALECT_LOSPN_PASSES_LOSPNPASSDETAILS_H

#include "mlir/Pass/Pass.h"
#include "LoSPN/LoSPNOps.h"

namespace mlir {
  namespace spn {
    namespace low {
#define GEN_PASS_CLASSES
#include "LoSPN/LoSPNPasses.h.inc"
    }
  }
}

#endif //SPNC_MLIR_LIB_DIALECT_LOSPN_PASSES_LOSPNPASSDETAILS_H
