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
