//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNINTERFACES_H
#define SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNINTERFACES_H

#include "LoSPN/LoSPNTraits.h"

namespace mlir {
namespace spn {
namespace low {
#include "LoSPN/LoSPNInterfaces.h.inc"
}
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNINTERFACES_H
