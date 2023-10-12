//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#pragma once

#include "IPUSubtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace spnc {

void initializeIPUTarget();

class IPUTargetMachine : public llvm::LLVMTargetMachine {
  IPUSubtarget subtarget_;

public:
  IPUTargetMachine(const llvm::Target &T, const llvm::Triple &TT, llvm::StringRef CPU, llvm::StringRef FS,
                   const llvm::TargetOptions &Options, llvm::Optional<llvm::Reloc::Model> RM,
                   llvm::Optional<llvm::CodeModel::Model> CM, llvm::CodeGenOpt::Level OL, bool JIT);

  virtual const llvm::TargetSubtargetInfo *getSubtargetImpl(const llvm::Function &) const override;

  
};
} // namespace spnc
