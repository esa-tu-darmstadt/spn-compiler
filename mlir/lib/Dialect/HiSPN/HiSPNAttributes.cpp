//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "HiSPN/HiSPNAttributes.h"

namespace mlir::spn::high {

// NOTE: Check if the actual verification of the input parameters happens on the DAG.
mlir::LogicalResult Bucket::verify(llvm::function_ref<mlir::InFlightDiagnostic ()>, int lb, int ub, llvm::APFloat val) {
    return mlir::success();
}

}

#define GET_ATTRDEF_CLASSES
#include "HiSPN/HiSPNAttributes.cpp.inc"
