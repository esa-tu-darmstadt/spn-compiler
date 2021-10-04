//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_LOSPNTOGPUPASSES_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_LOSPNTOGPUPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {

    std::unique_ptr<OperationPass<ModuleOp>> createGPUCopyEliminationPass();

    std::unique_ptr<OperationPass<FuncOp>> createGPUBufferDeallocationPass();

#define GEN_PASS_REGISTRATION
#include "LoSPNtoGPU/LoSPNtoGPUPasses.h.inc"

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_LOSPNTOGPUPASSES_H
