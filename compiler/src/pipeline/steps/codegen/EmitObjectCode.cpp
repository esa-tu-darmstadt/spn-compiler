//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "EmitObjectCode.h"
#include "option/Options.h"
#include "util/Logging.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Frontend/Driver/CodeGenOptions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

spnc::ExecutionResult spnc::EmitObjectCode::executeStep(llvm::Module *module,
                                                        ObjectFile *file) {
  std::error_code EC;
  llvm::raw_fd_ostream dest(file->fileName(), EC,
                            llvm::sys::fs::OpenFlags::OF_None);
  if (EC) {
    return spnc::failure("Could not open output file: {}", EC.message());
  }
  llvm::legacy::PassManager pass;

  // Add the target library info wrapper pass to the pass manager if available
  // This pass supplies information about the vector library to the codegen
  if (getContext()->has<llvm::TargetLibraryInfoImpl>()) {
    auto *TLII = getContext()->get<llvm::TargetLibraryInfoImpl>();
    pass.add(new llvm::TargetLibraryInfoWrapperPass(*TLII));
  }

  auto *machine = getContext()->get<llvm::TargetMachine>();
  auto fileType = llvm::CodeGenFileType::ObjectFile;

  if (machine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
    return spnc::failure("Cannot emit object file");
  }

  pass.run(*module);
  dest.flush();
  outFile = file;
  return success();
}

spnc::ObjectFile *spnc::EmitObjectCode::result() { return outFile; }