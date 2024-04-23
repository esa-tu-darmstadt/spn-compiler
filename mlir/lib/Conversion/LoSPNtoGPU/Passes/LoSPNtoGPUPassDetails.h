//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_LIB_CONVERSION_LOSPNTOGPU_PASSES_LOSPNTOGPUPASSDETAILS_H
#define SPNC_MLIR_LIB_CONVERSION_LOSPNTOGPU_PASSES_LOSPNTOGPUPASSDETAILS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace spn {
#define GEN_PASS_CLASSES
#include "LoSPNtoGPU/LoSPNtoGPUPasses.h.inc"
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_LIB_CONVERSION_LOSPNTOGPU_PASSES_LOSPNTOGPUPASSDETAILS_H
