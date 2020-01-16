//
// Created by ls on 1/15/20.
//

#include "LLVMWriteBitcode.h"
#include "llvm/Bitcode/BitcodeWriter.h"

namespace spnc {

    LLVMWriteBitcode::LLVMWriteBitcode(spnc::ActionWithOutput<llvm::Module> &_input, File<FileType::LLVM_BC> outputFile)
      : ActionSingleInput<llvm::Module, File<FileType::LLVM_BC> >{_input}, outFile{std::move(outputFile)} {}

    File<FileType::LLVM_BC> & LLVMWriteBitcode::execute() {
      if(!cached){
        std::error_code EC;
        llvm::raw_fd_ostream out_stream(outFile.fileName(), EC);
        llvm::WriteBitcodeToFile(input.execute(), out_stream);
        cached = true;
      }
      return outFile;
    }
}