//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LLVMLinker.h"
#include <util/Command.h>

using namespace spnc;

LLVMLinker::LLVMLinker(spnc::ActionWithOutput<LLVMBitcode>& _input1,
                       spnc::ActionWithOutput<LLVMBitcode>& _input2,
                       LLVMBitcode outputFile)
    : ActionDualInput<LLVMBitcode, LLVMBitcode, LLVMBitcode>{_input1, _input2},
      outFile{std::move(outputFile)} {}

LLVMBitcode& LLVMLinker::execute() {
  if (!cached) {
    // Invoke llvm-link as external command.
    std::vector<std::string> command;
    command.emplace_back("llvm-link");
    command.emplace_back("-o");
    command.push_back(outFile.fileName());
    command.push_back(input1.execute().fileName());
    command.push_back(input2.execute().fileName());
    Command::executeExternalCommand(command);
    cached = true;
  }
  return outFile;
}
