//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "IPUTargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

using namespace spnc;

llvm::Target &spnc::getTheIPUTarget() {
    static llvm::Target TheIPUTarget;
    return TheIPUTarget;
}

void spnc::initializeIPUTargetInfo() {
  llvm::RegisterTarget<llvm::Triple::UnknownArch> X(getTheIPUTarget(), "colossus",
                                    "GraphCore Colossus", "");
}