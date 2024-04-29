//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUCONVERSIONPASSES_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUCONVERSIONPASSES_H

#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace spn {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "LoSPNtoCPU/LoSPNtoCPUConversionPasses.h.inc"

} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUCONVERSIONPASSES_H
