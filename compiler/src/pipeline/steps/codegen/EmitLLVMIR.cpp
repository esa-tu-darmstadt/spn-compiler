//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "EmitLLVMIR.h"
#include "util/Logging.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <option/GlobalOptions.h>
#include <sstream>

spnc::ExecutionResult spnc::EmitLLVMIR::executeStep(llvm::Module *module, LLVMIR *ir) {
  // Emit module to file as IR
  std::error_code ec;
  llvm::raw_fd_ostream out(ir->fileName(), ec, llvm::sys::fs::OF_None);
  if (ec) {
    return spnc::failure("Could not open output file: {}", ec.message());
  }
  module->print(out, nullptr);
  out.flush();
  out.close();

  outFile = ir;
  return success();
}

spnc::LLVMIR *spnc::EmitLLVMIR::result() { return outFile; }