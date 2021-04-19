//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

#include "HiSPN/HiSPNDialect.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNPasses.h"
#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h"
#include "LoSPNtoCPU/LoSPNtoCPUConversionPasses.h"

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false));

static llvm::cl::opt<bool> cpuVectorize("cpu-vectorize",
                                        llvm::cl::desc("Vectorize code generated for CPU targets"),
                                        llvm::cl::init(false));

static llvm::cl::opt<bool> logSpace("use-log-space",
                                    llvm::cl::desc("Use log-space computation"),
                                    llvm::cl::init(false));

static llvm::cl::opt<bool> collectGraphStats("collect-graph-stats",
                                             llvm::cl::desc("Collect graph statistics"),
                                             llvm::cl::init(false));

static llvm::cl::opt<std::string> graphStatsFile{"graph-stats-file",
                                                 llvm::cl::desc("Graph statistics output file"),
                                                 llvm::cl::value_desc("filename"),
                                                 llvm::cl::init("/tmp/stats.json")};

///
/// spnc-opt: Custom tool to run SPN-dialect specific and generic passes on MLIR files.
int main(int argc, char** argv) {

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::spn::high::HiSPNDialect>();
  registry.insert<mlir::spn::low::LoSPNDialect>();

  mlir::registerAllPasses();
  mlir::spn::low::registerLoSPNPasses();

  mlir::registerPass("convert-hispn-query-to-lospn", "Convert queries from HiSPN to LoSPN dialect",
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createHiSPNtoLoSPNQueryConversionPass(logSpace);
                     });

  mlir::registerPass("convert-hispn-node-to-lospn", "Convert nodes from HiSPN to LoSPN dialect",
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createHiSPNtoLoSPNNodeConversionPass(logSpace);
                     });

  mlir::registerPass("convert-lospn-structure-to-cpu", "Convert structure from LoSPN to CPU target",
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createLoSPNtoCPUStructureConversionPass(cpuVectorize);
                     });

  mlir::registerPass("convert-lospn-nodes-to-cpu", "Convert nodes from LoSPN to CPU target",
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createLoSPNtoCPUNodeConversionPass();
                     });

  mlir::registerPass("vectorize-lospn-nodes", "Vectorize LoSPN nodes for CPU target",
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::createLoSPNNodeVectorizationPass();
                     });

  mlir::registerPass("collect-lospn-graph-stats", "Collect graph statistics",
                     []() -> std::unique_ptr<mlir::Pass> {
                       return mlir::spn::low::createLoSPNGraphStatsCollectionPass(graphStatsFile);
                     });

  llvm::InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR modular optimizer driver\n");

  mlir::MLIRContext context;
  if (showDialects) {
    llvm::outs() << "Registered Dialects:\n";
    for (mlir::Dialect* dialect : context.getLoadedDialects()) {
      llvm::outs() << dialect->getNamespace() << "\n";
    }
    return 0;
  }

  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << "Failed to open output file: " << errorMessage << "\n";
    exit(1);
  }

  if (failed(MlirOptMain(output->os(), std::move(file), passPipeline, registry,
                         splitInputFile, verifyDiagnostics, verifyPasses,
                         allowUnregisteredDialects))) {
    return 1;
  }
  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return 0;
}
