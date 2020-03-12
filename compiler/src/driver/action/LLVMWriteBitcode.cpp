//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LLVMWriteBitcode.h"
#include "llvm/Bitcode/BitcodeWriter.h"

using namespace spnc;

LLVMWriteBitcode::LLVMWriteBitcode(spnc::ActionWithOutput<llvm::Module>& _input, LLVMBitcode outputFile)
    : ActionSingleInput<llvm::Module, LLVMBitcode>{_input}, outFile{std::move(outputFile)} {}

File<FileType::LLVM_BC>& LLVMWriteBitcode::execute() {
  if (!cached) {
    // Use WriteBitCode API to write module to file.
    std::error_code EC;
    llvm::raw_fd_ostream out_stream(outFile.fileName(), EC);
    llvm::WriteBitcodeToFile(input.execute(), out_stream);
    cached = true;
  }
  return outFile;
}
