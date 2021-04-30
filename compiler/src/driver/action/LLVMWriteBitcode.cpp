//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

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
