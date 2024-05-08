//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "HiSPN/HiSPNDialect.h"
#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h"
#include "HiSPNtoLoSPN/HiSPNtoLoSPNPipeline.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNPasses.h"
#include "LoSPNtoCPU/LoSPNtoCPUConversionPasses.h"
#include "LoSPNtoCPU/LoSPNtoCPUPipeline.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#if SPNC_CUDA_SUPPORT
#include "LoSPNtoGPU/LoSPNtoGPUPasses.h"
#endif

static llvm::cl::opt<std::string> graphStatsFile{
    "graph-stats-file", llvm::cl::desc("Graph statistics output file"),
    llvm::cl::value_desc("filename"), llvm::cl::init("/tmp/stats.json")};

///
/// spnc-opt: Custom tool to run SPN-dialect specific and generic passes on MLIR
/// files.
int main(int argc, char **argv) {

  mlir::registerAllPasses();
  mlir::spn::low::registerLoSPNPasses();
  mlir::spn::registerLoSPNtoCPUPasses();
  mlir::spn::registerHiSPNtoLoSPNPasses();
#if SPNC_CUDA_SUPPORT
  mlir::spn::registerLoSPNtoGPUPasses();
#endif

  mlir::spn::registerHiSPNtoLoSPNPipeline();
  mlir::spn::registerLoSPNtoCPUPipeline();

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::spn::low::createLoSPNGraphStatsCollectionPass(graphStatsFile);
  });

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::spn::high::HiSPNDialect>();
  registry.insert<mlir::spn::low::LoSPNDialect>();

  return failed(
      mlir::MlirOptMain(argc, argv, "SPNC optimizer driver\n", registry));
}
