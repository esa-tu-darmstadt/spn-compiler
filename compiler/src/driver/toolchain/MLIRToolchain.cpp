//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <util/FileSystem.h>
#include <driver/GlobalOptions.h>
#include <HiSPN/HiSPNDialect.h>
#include <LoSPN/LoSPNDialect.h>
#include "MLIRToolchain.h"
#include "mlir/InitAllDialects.h"
#include <llvm/ADT/StringMap.h>
#include <llvm/MC/SubtargetFeature.h>
#include <llvm/Support/TargetSelect.h>
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

using namespace spnc;
using namespace mlir;


void spnc::MLIRToolchain::initializeMLIRContext(mlir::MLIRContext& ctx) {
  DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::spn::high::HiSPNDialect>();
  registry.insert<mlir::spn::low::LoSPNDialect>();
  ctx.loadDialect<mlir::spn::high::HiSPNDialect>();
  ctx.loadDialect<mlir::spn::low::LoSPNDialect>();
  ctx.loadDialect<mlir::StandardOpsDialect>();
  ctx.loadDialect<mlir::scf::SCFDialect>();
  ctx.loadDialect<mlir::memref::MemRefDialect>();
  ctx.loadDialect<mlir::LLVM::LLVMDialect>();
  ctx.loadDialect<mlir::vector::VectorDialect>();
  ctx.loadDialect<mlir::math::MathDialect>();
  ctx.loadDialect<mlir::gpu::GPUDialect>();
  ctx.loadDialect<mlir::NVVM::NVVMDialect>();
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
    spdlog::level::level_enum level = spdlog::level::level_enum::debug;
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
  std::stringstream featureList;
  bool initial = true;
  if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
    for (auto& f : hostFeatures) {
      // Temporary hack: If no vectorization was requested by the user, disable
      // AVX* target features to avoid the LLVM auto-vectorizer to kick in.
      // TODO: Replace with a cleaner solution.
      if (!cpuVectorize && f.first().startswith("avx")) {
        features.AddFeature(f.first(), false);
      } else {
        features.AddFeature(f.first(), f.second);
        if(f.second){
          if (!initial) {
            featureList << ", ";
          }
          featureList << f.first().str();
          initial = false;
        }
      }
    }
  }
  SPDLOG_INFO("Target machine default triple: {}", targetTriple);
  SPDLOG_INFO("Target machine CPU name: {}", cpu);
  SPDLOG_INFO("Target machine features: {}", featureList.str());
  SPDLOG_INFO("Target machine CPU physical core count: {}", llvm::sys::getHostNumPhysicalCores());

  std::shared_ptr<llvm::TargetMachine> machine{target->createTargetMachine(targetTriple,
                                                                           cpu, features.getString(), {},
                                                                           llvm::Reloc::PIC_)};
  return machine;
}

llvm::SmallVector<std::string> spnc::MLIRToolchain::parseLibrarySearchPaths(const std::string& paths){
  llvm::SmallVector<std::string> searchPaths;
  std::istringstream tokenStream(paths);
  std::string token;
  while(std::getline(tokenStream, token, ':')) {
    searchPaths.push_back(token);
  }
  return searchPaths;
}