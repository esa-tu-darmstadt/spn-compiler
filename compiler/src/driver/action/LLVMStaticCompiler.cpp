//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LLVMStaticCompiler.h"
#include <util/Command.h>

using namespace spnc;

LLVMStaticCompiler::LLVMStaticCompiler(spnc::ActionWithOutput<LLVMBitcode>& _input,
                                       ObjectFile outputFile)
    : ActionSingleInput<LLVMBitcode, ObjectFile>{_input},
      outFile{std::move(outputFile)} {}

ObjectFile& LLVMStaticCompiler::execute() {
  if (!cached) {
    // Invoke llc as external command.
    std::vector<std::string> command;
    command.emplace_back("llc");
    command.emplace_back("--relocation-model=pic");
    command.emplace_back("--filetype=obj");
    command.emplace_back("-O3");
    command.emplace_back("-o");
    command.push_back(outFile.fileName());
    command.push_back(input.execute().fileName());
    Command::executeExternalCommand(command);
    cached = true;
  }
  return outFile;
}
