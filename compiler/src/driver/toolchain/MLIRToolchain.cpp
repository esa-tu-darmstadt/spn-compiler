//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <util/FileSystem.h>
#include <driver/BaseActions.h>
#include <driver/GlobalOptions.h>
#include <SPN/SPNDialect.h>
#include <HiSPN/HiSPNDialect.h>
#include <LoSPN/LoSPNDialect.h>
#include <codegen/mlir/pipeline/SPNDialectPipeline.h>
#include "codegen/mlir/conversion/HiSPNtoLoSPNConversion.h"
#include "codegen/mlir/conversion/LoSPNtoCPUConversion.h"
#include "codegen/mlir/conversion/CPUtoLLVMConversion.h"
#include "codegen/mlir/conversion/MLIRtoLLVMIRConversion.h"
#include "codegen/mlir/analysis/CollectGraphStatistics.h"
#include <driver/action/ClangKernelLinking.h>
#include <codegen/mlir/frontend/MLIRDeserializer.h>
#include "MLIRToolchain.h"
#include "mlir/InitAllDialects.h"
#include <llvm/ADT/StringMap.h>
#include <llvm/MC/SubtargetFeature.h>
#include <llvm/Support/TargetSelect.h>
#include <driver/action/EmitObjectCode.h>
#include <codegen/mlir/transformation/LoSPNTransformations.h>
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "mlir/Target/LLVMIR.h"

using namespace spnc;

std::unique_ptr<Job<Kernel> > MLIRToolchain::constructJobFromFile(const std::string& inputFile,
                                                                  std::shared_ptr<Configuration> config) {
  // Uncomment the following two lines to get detailed output during MLIR dialect conversion;
  //llvm::DebugFlag = true;
  //llvm::setCurrentDebugType("dialect-conversion");
  std::unique_ptr<Job<Kernel>> job = std::make_unique<Job<Kernel>>(config);
  // Invoke MLIR code-generation on parsed tree.
  auto ctx = std::make_shared<MLIRContext>();
  initializeMLIRContext(*ctx);
  // If IR should be dumped between steps/passes, we need to disable
  // multi-threading in MLIR
  if (spnc::option::dumpIR.get(*config)) {
    ctx->enableMultithreading(false);
  }
  auto diagHandler = setupDiagnosticHandler(ctx.get());
  auto cpuVectorize = spnc::option::cpuVectorize.get(*config);
  SPDLOG_INFO("CPU Vectorization enabled: {}", cpuVectorize);
  auto targetMachine = createTargetMachine(cpuVectorize);
  auto kernelInfo = std::make_shared<KernelInfo>();
  BinarySPN binarySPNFile{inputFile, false};
  auto& deserialized = job->insertAction<MLIRDeserializer>(std::move(binarySPNFile), ctx, kernelInfo);
  auto& hispn2lospn = job->insertAction<HiSPNtoLoSPNConversion>(deserialized, ctx, diagHandler);
  // If requested via the configuration, collect graph statistics.
  // TODO: Graph statistics collection is currently disabled, as it does not yet work
  // with the LoSPN dialect.
  // TODO: Move this to LoSPNTransformations
  if (false && spnc::option::collectGraphStats.get(*config)) {
    auto deleteTmps = spnc::option::deleteTemporaryFiles.get(*config);
    // Collect graph statistics on transformed / canonicalized MLIR.
    auto statsFile = StatsFile(spnc::option::graphStatsFile.get(*config), deleteTmps);
    auto& graphStats = job->insertAction<CollectGraphStatistics>(hispn2lospn, std::move(statsFile));
    // Join the two actions happening on the transformed module (Graph-Stats & SPN-to-Standard-MLIR lowering).
    auto& joinAction = job->insertAction<JoinAction<mlir::ModuleOp, StatsFile>>(hispn2lospn, graphStats);
    //spnPipelineResult = &joinAction;
  }
  auto& lospnTransform = job->insertAction<LoSPNTransformations>(hispn2lospn, ctx, diagHandler, kernelInfo);
  auto& lospn2cpu = job->insertAction<LoSPNtoCPUConversion>(lospnTransform, ctx, diagHandler);
  auto& cpu2llvm = job->insertAction<CPUtoLLVMConversion>(lospn2cpu, ctx, diagHandler);

  // Convert the MLIR module to a LLVM-IR module.
  auto& llvmConversion = job->insertAction<MLIRtoLLVMIRConversion>(cpu2llvm, ctx, targetMachine);

  // Translate the generated LLVM IR module to object code and write it to an object file.
  auto objectFile = FileSystem::createTempFile<FileType::OBJECT>(false);
  SPDLOG_INFO("Generating object file {}", objectFile.fileName());
  auto& emitObjectCode = job->insertAction<EmitObjectCode>(llvmConversion, std::move(objectFile), targetMachine);

  // Link generated object file into shared object.
  auto sharedObject = FileSystem::createTempFile<FileType::SHARED_OBJECT>(false);
  SPDLOG_INFO("Compiling to shared object file {}", sharedObject.fileName());
  auto& linkSharedObject =
      job->insertFinalAction<ClangKernelLinking>(emitObjectCode, std::move(sharedObject), kernelInfo);
  return std::move(job);
}

void spnc::MLIRToolchain::initializeMLIRContext(mlir::MLIRContext& ctx) {
  DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::spn::high::HiSPNDialect>();
  registry.insert<mlir::spn::low::LoSPNDialect>();
  ctx.loadDialect<mlir::spn::high::HiSPNDialect>();
  ctx.loadDialect<mlir::spn::low::LoSPNDialect>();
  ctx.loadDialect<mlir::StandardOpsDialect>();
  ctx.loadDialect<mlir::scf::SCFDialect>();
  ctx.loadDialect<mlir::LLVM::LLVMDialect>();
  ctx.loadDialect<mlir::vector::VectorDialect>();
  ctx.loadDialect<mlir::math::MathDialect>();
  ctx.loadDialect<mlir::linalg::LinalgDialect>();
  ctx.appendDialectRegistry(registry);
  mlir::registerLLVMDialectTranslation(ctx);
  for (auto* D : ctx.getLoadedDialects()) {
    SPDLOG_INFO("Loaded dialect: {}", D->getNamespace().str());
  }
}

std::shared_ptr<mlir::ScopedDiagnosticHandler> spnc::MLIRToolchain::setupDiagnosticHandler(mlir::MLIRContext* ctx) {
  // Create a simple diagnostic handler that will forward the diagnostic information to the SPDLOG instance
  // used by the compiler/toolchain.
  return std::make_shared<mlir::ScopedDiagnosticHandler>(ctx, [](Diagnostic& diag) {
    auto logger = spdlog::default_logger_raw();
    spdlog::level::level_enum level;
    std::string levelTxt;
    // Translate from MLIR severity to SPDLOG log-level.
    switch (diag.getSeverity()) {
      case DiagnosticSeverity::Note: level = spdlog::level::level_enum::debug;
        levelTxt = "NOTE";
        break;
      case DiagnosticSeverity::Remark: level = spdlog::level::level_enum::info;
        levelTxt = "REMARK";
        break;
      case DiagnosticSeverity::Warning: level = spdlog::level::level_enum::warn;
        levelTxt = "WARNING";
        break;
      case DiagnosticSeverity::Error: level = spdlog::level::level_enum::err;
        levelTxt = "ERROR";
        break;
    }
    // Also emit all notes with log-level "trace", as they can be very verbose.
    logger->log(level, "MLIR {}: {}", levelTxt, diag.str());
    for (auto& n : diag.getNotes()) {
      logger->log(spdlog::level::level_enum::trace, n.str());
    }
    return success();
  });
}

std::shared_ptr<llvm::TargetMachine> spnc::MLIRToolchain::createTargetMachine(bool cpuVectorize) {
  // NOTE: If we wanted to support cross-compilation, we could hook in here to use a different target machine.

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target) {
    SPNC_FATAL_ERROR("No target for target triple {}: {}", targetTriple, errorMessage);
  }
  std::string cpu{llvm::sys::getHostCPUName()};
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;
  if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
    for (auto& f : hostFeatures) {
      // Temporary hack: If no vectorization was requested by the user, disable
      // AVX* target features to avoid the LLVM auto-vectorizer to kick in.
      // TODO: Replace with a cleaner solution.
      if (!cpuVectorize && f.first().startswith("avx")) {
        features.AddFeature(f.first(), false);
      } else {
        features.AddFeature(f.first(), f.second);
      }
    }
  }
  SPDLOG_INFO("Target machine CPU name: {}", cpu);
  SPDLOG_INFO("Target machine features: {}", features.getString());

  std::shared_ptr<llvm::TargetMachine> machine{target->createTargetMachine(targetTriple,
                                                                           cpu, features.getString(), {},
                                                                           llvm::Reloc::PIC_)};
  return std::move(machine);
}
