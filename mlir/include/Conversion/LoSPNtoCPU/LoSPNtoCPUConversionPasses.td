//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef MLIR_CONVERSION_LOSPNTOCPUCONVERSIONPASSES
#define MLIR_CONVERSION_LOSPNTOCPUCONVERSIONPASSES

include "mlir/Pass/PassBase.td"

def LoSPNNodeVectorizationPass : Pass<"vectorize-lospn-nodes", "ModuleOp"> {
  let summary = "Vectorize LoSPN nodes for CPU target";
  let dependentDialects = ["arith::ArithDialect", "scf::SCFDialect", "math::MathDialect", "vector::VectorDialect", "memref::MemRefDialect"];
}

def LoSPNtoCPUStructureConversionPass : Pass<"convert-lospn-structure-to-cpu", "ModuleOp"> {
  let summary = "Convert structure from LoSPN to CPU target";
  let dependentDialects = ["arith::ArithDialect", "scf::SCFDialect", "math::MathDialect", "vector::VectorDialect", "memref::MemRefDialect", "func::FuncDialect"];

  let options = [
    Option<"vectorize", "cpu-vectorize", "bool", "false", "Vectorize code generated for CPU targets">,
    Option<"maxAttempts", "slp-max-attempts", "unsigned", "1", "Maximum number of SLP vectorization attempts">,
    Option<"maxSuccessfulIterations", "slp-max-successful-iterations", "unsigned", "1", "Maximum number of successful SLP vectorization runs to be applied to a function">,
    Option<"maxNodeSize", "slp-max-node-size", "unsigned", "10", "Maximum multinode size during SLP vectorization in terms of the number of vectors they may contain">,
    Option<"maxLookAhead", "slp-max-look-ahead", "unsigned", "3", "Maximum look-ahead depth when reordering multinode operands during SLP vectorization">,
    Option<"reorderInstructionsDFS", "slp-reorder-instructions-dfs", "bool", "true", "Flag to indicate if SLP-vectorized instructions should be arranged in DFS order (true) or in BFS order (false)">,
    Option<"allowDuplicateElements", "slp-allow-duplicate-elements", "bool", "false", "Flag to indicate whether duplicate elements are allowed in vectors during SLP graph building">,
    Option<"allowTopologicalMixing", "slp-allow-topological-mixing", "bool", "false", "Flag to indicate if elements with different topological depths are allowed in vectors during SLP graph building">,
    Option<"useXorChains", "slp-use-xor-chains", "bool", "true", "Flag to indicate if XOR chains should be used to compute look-ahead scores instead of Porpodas's algorithm">
  ];
}

def LoSPNtoCPUNodeConversionPass : Pass<"convert-lospn-nodes-to-cpu", "ModuleOp"> {
  let summary = "Convert nodes from LoSPN to CPU target";
  let dependentDialects = ["arith::ArithDialect", "scf::SCFDialect", "math::MathDialect", "vector::VectorDialect", "memref::MemRefDialect", "func::FuncDialect"];
}

#endif // MLIR_CONVERSION_LOSPNTOCPUCONVERSIONPASSES