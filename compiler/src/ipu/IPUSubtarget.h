//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

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