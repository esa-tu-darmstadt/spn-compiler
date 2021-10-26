//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

#include "HiSPN/HiSPNDialect.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNPasses.h"
#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h"
#include "LoSPNtoCPU/LoSPNtoCPUConversionPasses.h"
#if SPNC_CUDA_SUPPORT
#include "LoSPNtoGPU/LoSPNtoGPUPasses.h"
#endif

static llvm::cl::opt<bool> cpuVectorize("cpu-vectorize",
                                        llvm::cl::desc("Vectorize code generated for CPU targets"),
                                        llvm::cl::init(false));

static llvm::cl::opt<unsigned> slpMaxAttempts
    ("slp-max-attempts", llvm::cl::desc("Maximum number of SLP vectorization attempts"), llvm::cl::init(1));

static llvm::cl::opt<unsigned> slpMaxSuccessfulIterations("slp-max-successful-iterations",
                                                          llvm::cl::desc(
                                                              "Maximum number of successful SLP vectorization runs to be applied to a function"),
                                                          llvm::cl::init(1));

static llvm::cl::opt<unsigned> slpMaxNodeSize("slp-max-node-size",
                                              llvm::cl::desc(
                                                  "Maximum multinode size during SLP vectorization in terms of the number of vectors they may contain"),
                                              llvm::cl::init(10));

static llvm::cl::opt<unsigned> slpMaxLookAhead("slp-max-look-ahead",
                                               llvm::cl::desc(
                                                   "Maximum look-ahead depth when reordering multinode operands during SLP vectorization"),
                                               llvm::cl::init(3));

static llvm::cl::opt<bool> slpReorderInstructionsDFS("slp-reorder-dfs",
                                                     llvm::cl::desc(
                                                         "Flag to indicate the order in which SLP-vectorized instructions should be arranged"),
                                                     llvm::cl::init(true));

static llvm::cl::opt<bool> slpAllowDuplicateElements("slp-allow-duplicate-elements",
                                                     llvm::cl::desc(
                                                         "Flag to indicate whether duplicate elements are allowed in vectors during SLP graph building"),
                                                     llvm::cl::init(false));

static llvm::cl::opt<bool> slpAllowTopologicalMixing("slp-allow-topological-mixing",
                                                     llvm::cl::desc(
                                                         "Flag to indicate if elements with different topological depths are allowed in vectors during SLP graph building"),
                                                     llvm::cl::init(false));

static llvm::cl::opt<bool> slpUseXorChains("slp-use-xor-chains",
                                           llvm::cl::desc(
                                               "Flag to indicate if XOR chains should be used to compute look-ahead scores instead of Porpodas's algorithm"),
                                           llvm::cl::init(false));

static llvm::cl::opt<bool> logSpace("use-log-space",
                                    llvm::cl::desc("Use log-space computation"),
                                    llvm::cl::init(false));

static llvm::cl::opt<bool> optRepresentation("opt-repr",
                                             llvm::cl::desc("Determine and use optimal number representation"),
                                             llvm::cl::init(false));

static llvm::cl::opt<std::string> graphStatsFile{"graph-stats-file",
                                                 llvm::cl::desc("Graph statistics output file"),
                                                 llvm::cl::value_desc("filename"),
                                                 llvm::cl::init("/tmp/stats.json")};

///
/// spnc-opt: Custom tool to run SPN-dialect specific and generic passes on MLIR files.
int main(int argc, char** argv) {

  mlir::registerAllPasses();
  mlir::spn::low::registerLoSPNPasses();
#if SPNC_CUDA_SUPPORT
  mlir::spn::registerLoSPNtoGPUPasses();
#endif

  mlir::registerPass("convert-hispn-query-to-lospn", "Convert queries from HiSPN to LoSPN dialect",
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createHiSPNtoLoSPNQueryConversionPass(logSpace, optRepresentation);
                     });

  mlir::registerPass("convert-hispn-node-to-lospn", "Convert nodes from HiSPN to LoSPN dialect",
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createHiSPNtoLoSPNNodeConversionPass(logSpace, optRepresentation);
                     });

  mlir::registerPass("convert-lospn-structure-to-cpu", "Convert structure from LoSPN to CPU target",
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createLoSPNtoCPUStructureConversionPass(cpuVectorize,
                                                                                 slpMaxAttempts,
                                                                                 slpMaxSuccessfulIterations,
                                                                                 slpMaxNodeSize,
                                                                                 slpMaxLookAhead,
                                                                                 slpReorderInstructionsDFS,
                                                                                 slpAllowDuplicateElements,
                                                                                 slpAllowTopologicalMixing,
                                                                                 slpUseXorChains);
                     });

  mlir::registerPass("convert-lospn-nodes-to-cpu", "Convert nodes from LoSPN to CPU target",
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createLoSPNtoCPUNodeConversionPass();
                     });

  mlir::registerPass("vectorize-lospn-nodes", "Vectorize LoSPN nodes for CPU target",
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createLoSPNNodeVectorizationPass();
                     });

  mlir::registerPass("collect-graph-stats", "Collect graph statistics",
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::low::createLoSPNGraphStatsCollectionPass(graphStatsFile);
                     });

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::spn::high::HiSPNDialect>();
  registry.insert<mlir::spn::low::LoSPNDialect>();

  return failed(
      mlir::MlirOptMain(argc, argv, "SPNC optimizer driver\n", registry, false));
}
