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
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "HiSPN/HiSPNDialect.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNPasses.h"
#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h"
#include "LoSPNtoCPU/LoSPNtoCPUConversionPasses.h"
#if SPNC_CUDA_SUPPORT
#include "LoSPNtoGPU/LoSPNtoGPUPasses.h"
#endif
#include "LoSPNtoFPGA/LoSPNtoFPGAPass.h"

#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"

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

  mlir::registerPass(
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createHiSPNtoLoSPNQueryConversionPass(logSpace, optRepresentation);
                     });

  mlir::registerPass(
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createHiSPNtoLoSPNNodeConversionPass(logSpace, optRepresentation);
                     });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<mlir::spn::LoSPNtoCPUStructureConversionPass>();
  });

  mlir::registerPass(
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createLoSPNtoCPUNodeConversionPass();
                     });

  mlir::registerPass(
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createLoSPNNodeVectorizationPass();
                     });

  mlir::registerPass(
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::low::createLoSPNGraphStatsCollectionPass(graphStatsFile);
                     });

  mlir::registerPass(
    []() -> std::unique_ptr<mlir::Pass> {
      return mlir::spn::fpga::createLoSPNtoFPGAPass();
    }
  );

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::spn::high::HiSPNDialect>();
  registry.insert<mlir::spn::low::LoSPNDialect>();

  // circt stuff
  //circt::registerAllPasses();
  circt::registerExportVerilogPass();
  circt::registerExportSplitVerilogPass();
  registry.insert<circt::hw::HWDialect,
                  circt::seq::SeqDialect,
                  circt::sv::SVDialect>();
  mlir::registerPass(
    circt::seq::createSeqLowerToSVPass
  );
  mlir::registerPass(
    []() { return circt::seq::createSeqFIRRTLLowerToSVPass(); }
  );

  return failed(
      mlir::MlirOptMain(argc, argv, "SPNC optimizer driver\n", registry, false));
}
