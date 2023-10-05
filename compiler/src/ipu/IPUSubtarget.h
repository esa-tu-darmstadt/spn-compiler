#pragma once

#include "ipu/IPUTargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
namespace spnc {

class IPUSubtarget : public llvm::TargetSubtargetInfo {
  IPUTargetLowering TLInfo;

public:
  explicit IPUSubtarget(const llvm::Triple &TT, llvm::StringRef CPU, llvm::StringRef TuneCPU, llvm::StringRef FS,
                        const llvm::TargetMachine &TM);

  const IPUTargetLowering *getTargetLowering() const override { return &TLInfo; }
};
} // namespace spnc