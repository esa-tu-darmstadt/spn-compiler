//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "EmitObjectCodeForIPU.h"
#include "ipu/IPUTargetMachine.h"
#include "util/FileSystem.h"
#include "util/Logging.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <option/GlobalOptions.h>
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/TargetRegistry.h"
#include <sstream>

using namespace spnc;

template <FileType SourceType>
spnc::ExecutionResult spnc::EmitObjectCodeForIPU<SourceType>::executeStep(File<SourceType> *source,
                                                                          CompiledGraphProgram *graphProgram) {
  // Invoke the popc compiler
  auto &config = *this->getContext()->template get<Configuration>();
  std::string popcPath = spnc::option::ipuCompilerPath.get(config);
  int optimizationLevel = spnc::option::optLevel.get(config);
  llvm::TargetMachine &targetMachine = *this->getContext()->template get<llvm::TargetMachine>();

  std::stringstream popcArgs;
  if(targetMachine.getTargetTriple().getArchName() == "colossus") {
    popcArgs << " -target " << targetMachine.getTargetCPU().str();
  } else {
    popcArgs << " -target cpu";
  }

  popcArgs << " -O" << optimizationLevel;
  popcArgs << " -o " << graphProgram->fileName();
  popcArgs << " --input-file " << source->fileName();
  popcArgs << " -X -Qunused-arguments";

  std::string popcCmd = popcPath + popcArgs.str();

  SPDLOG_INFO("Invoking IPU compiler: {}", popcCmd);

  int popcRet = std::system(popcCmd.c_str());
  if (popcRet != 0) {
    return spnc::failure("The IPU compiler returned with non-zero exit code: {}", popcRet);
  }
  return success();
}

// Explicitly instantiate the template for the supported source types
template class spnc::EmitObjectCodeForIPU<FileType::CPP>;
template class spnc::EmitObjectCodeForIPU<FileType::LLVM_IR>;