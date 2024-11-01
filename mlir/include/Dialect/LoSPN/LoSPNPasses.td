//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef MLIR_DIALECT_LoSPN_PASSES
#define MLIR_DIALECT_LoSPN_PASSES

include "mlir/Pass/PassBase.td"

def LoSPNBufferize : Pass<"lospn-bufferize", "ModuleOp"> {
  let summary = "Bufferize LoSPN operations";
  let dependentDialects = ["memref::MemRefDialect", "bufferization::BufferizationDialect"];
}

def LoSPNCopyRemoval : Pass<"lospn-copy-removal", "SPNKernel"> {
  let summary = "Remove unnecessary copy operations in the LoSPN dialect";
  let dependentDialects = ["memref::MemRefDialect", "scf::SCFDialect"];
}

def LoSPNTaskPartioning : Pass<"lospn-task-partitioning", "SPNKernel"> {
  let summary = "Partition LoSPN tasks into multiple tasks";
  let options = [
    Option<"maxTaskSize", "max-task-size", "int", "-1", "Determine the maximum number of operations per task">,
    Option<"decomposeTaskInputs", "decompose-task-inputs", "bool", "false", "If enabled, each task receives only the part of the input tensor that it actually uses">,
    Option<"schedule", "schedule", "bool", "false", "If enabled, the partitions are scheduled and the schedule is attached as an attribute to the kernel.">
  ];
}

#endif // MLIR_DIALECT_LoSPN_PASSES