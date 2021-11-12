//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "EmitObjectCode.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "util/Logging.h"
#include "llvm/IR/LegacyPassManager.h"
#include <option/GlobalOptions.h>
#include "llvm/Analysis/TargetLibraryInfo.h"

spnc::ExecutionResult spnc::EmitObjectCode::executeStep(llvm::Module* module, ObjectFile* file) {
  std::error_code EC;
  llvm::raw_fd_ostream dest(file->fileName(), EC, llvm::sys::fs::OpenFlags::OF_None);
  if (EC) {
    return spnc::failure("Could not open output file: {}", EC.message());
  }
  llvm::legacy::PassManager pass;
  // If a vector library was specified via the CLI option, add the functions from the vector
  // library to the TargetLibraryInfo and add a TargetLibraryInfoWrapperPass to the pass manager
  // to automatically replace vectorized calls to functions such as exp or log with optimized
  // functions from the vector library.
  auto veclib = spnc::option::vectorLibrary.get(*getContext()->get<Configuration>());
  auto* machine = getContext()->get<llvm::TargetMachine>();
  if (veclib != spnc::option::VectorLibrary::NONE) {
    llvm::TargetLibraryInfoImpl TLII(llvm::Triple(machine->getTargetTriple()));
    switch (veclib) {
      case spnc::option::VectorLibrary::SVML:TLII.addVectorizableFunctionsFromVecLib(llvm::TargetLibraryInfoImpl::SVML);
        break;
      case spnc::option::VectorLibrary::LIBMVEC:TLII.addVectorizableFunctionsFromVecLib(llvm::TargetLibraryInfoImpl::LIBMVEC_X86);
        break;
      case spnc::option::VectorLibrary::ARM: /* ARM Optimized Routines are not available through the TLII.*/ break;
      default: return spnc::failure("Unknown vector library");
    }
    pass.add(new llvm::TargetLibraryInfoWrapperPass(TLII));
  }
  auto fileType = llvm::CGFT_ObjectFile;

  if (machine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
    return spnc::failure("Cannot emit object file");
  }

  pass.run(*module);
  dest.flush();
  outFile = file;
  return success();
}

spnc::ObjectFile* spnc::EmitObjectCode::result() {
  return outFile;
}