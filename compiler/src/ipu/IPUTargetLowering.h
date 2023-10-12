//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

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