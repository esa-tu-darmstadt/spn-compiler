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
  let constructor = "mlir::spn::low::createLoSPNBufferizePass()";
  let dependentDialects = ["memref::MemRefDialect", "arith::ArithDialect", "bufferization::BufferizationDialect"];
}

def LoSPNCopyRemoval : Pass<"lospn-copy-removal", "SPNKernel"> {
  let summary = "Remove unnecessary copy operations in the LoSPN dialect";
  let constructor = "mlir::spn::low::createLoSPNCopyRemovalPass()";
  let dependentDialects = ["memref::MemRefDialect", "arith::ArithDialect", "scf::SCFDialect"];
}

def LoSPNTaskPartioning : Pass<"lospn-task-partitioning", "SPNKernel"> {
  let summary = "Partition LoSPN tasks into multiple tasks";
  let constructor = "mlir::spn::low::createLoSPNPartitionerPass()";
  let options = [
    Option<"maxTaskSize", "max-task-size", "int", "-1", "Determine the maximum number of operations per task">,
  ];
}

def ReplaceARMOptimizedRoutines : Pass<"replace-arm-optimized-routines", "ModuleOp"> {
  let summary = "Replace elementary functions with calls to ARM optimized routines";
  let constructor = "mlir::spn::low::createReplaceARMOptimizedRoutinesPass()";
}

#endif // MLIR_DIALECT_LoSPN_PASSES