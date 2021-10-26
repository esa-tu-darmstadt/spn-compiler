//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef MLIR_DIALECT_LoSPNtoGPU_PASSES
#define MLIR_DIALECT_LoSPNtoGPU_PASSES

include "mlir/Pass/PassBase.td"

def GPUCopyElimination : Pass<"gpu-copy-elimination", "ModuleOp"> {
  let summary = "Eliminate unnecessary data transfers between host and GPU device";
  let constructor = "mlir::spn::createGPUCopyEliminationPass()";
}

def GPUBufferDeallocation : Pass<"gpu-buffer-deallocation", "FuncOp"> {
  let summary = "Insert de-allocation for buffers in GPU memory";
  let constructor = "mlir::spn::createGPUBufferDeallocationPass()";
}

#endif // MLIR_DIALECT_LoSPNtoGPU_PASSES