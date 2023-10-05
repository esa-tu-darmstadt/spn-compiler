#pragma once

#include "llvm/CodeGen/TargetLowering.h"

namespace spnc {
class IPUSubtarget;

class IPUTargetLowering : public llvm::TargetLowering {
  const IPUSubtarget &Subtarget;

public:
  explicit IPUTargetLowering(const llvm::TargetMachine &TM, const IPUSubtarget &STI);
};
} // namespace spnc