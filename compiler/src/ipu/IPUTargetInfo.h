//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#pragma once

namespace llvm {
class Target;
}

namespace spnc {
llvm::Target &getTheIPUTarget();
void initializeIPUTargetInfo();
}