//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "EmitObjectCode.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "util/Logging.h"
#include "llvm/IR/LegacyPassManager.h"
#include <driver/GlobalOptions.h>
#include "llvm/Analysis/TargetLibraryInfo.h"

spnc::EmitObjectCode::EmitObjectCode(ActionWithOutput<llvm::Module>& _module,
                                     ObjectFile outputFile,
                                     std::shared_ptr<llvm::TargetMachine> targetMachine)
    : ActionSingleInput<llvm::Module, ObjectFile>(_module),
      outFile{std::move(outputFile)}, machine{std::move(targetMachine)} {}

spnc::ObjectFile& spnc::EmitObjectCode::execute() {
  if (!cached) {
    std::error_code EC;
    llvm::raw_fd_ostream dest(outFile.fileName(), EC, llvm::sys::fs::OpenFlags::OF_None);
    if (EC) {
      SPNC_FATAL_ERROR("Could not open output file: {}", EC.message());
    }
    llvm::legacy::PassManager pass;
    // If a vector library was specified via the CLI option, add the functions from the vector
    // library to the TargetLibraryInfo and add a TargetLibraryInfoWrapperPass to the pass manager
    // to automatically replace vectorized calls to functions such as exp or log with optimized
    // functions from the vector library.
    auto veclib = spnc::option::vectorLibrary.get(*config);
    if (veclib != spnc::option::VectorLibrary::NONE) {
      llvm::TargetLibraryInfoImpl TLII(llvm::Triple(machine->getTargetTriple()));
      switch (veclib) {
        case spnc::option::VectorLibrary::SVML:TLII.addVectorizableFunctionsFromVecLib(llvm::TargetLibraryInfoImpl::SVML);
          break;
        case spnc::option::VectorLibrary::LIBMVEC:TLII.addVectorizableFunctionsFromVecLib(llvm::TargetLibraryInfoImpl::LIBMVEC_X86);
          break;
        default:SPNC_FATAL_ERROR("Unknown vector library");
      }
      pass.add(new llvm::TargetLibraryInfoWrapperPass(TLII));
    }
    auto fileType = llvm::CGFT_ObjectFile;

    if (machine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
      SPNC_FATAL_ERROR("Cannot emit object file");
    }

    pass.run(input.execute());
    dest.flush();
  }
  return outFile;
}
