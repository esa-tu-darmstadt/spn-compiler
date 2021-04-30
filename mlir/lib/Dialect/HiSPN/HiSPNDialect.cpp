//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "HiSPN/HiSPNDialect.h"
#include "HiSPN/HiSPNOps.h"
#include "HiSPN/HiSPNAttributes.h"
#include "HiSPN/HiSPNInterfaces.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::spn::high;

void HiSPNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "HiSPN/HiSPNOps.cpp.inc"
  >();
  addTypes<ProbabilityType>();
}

::mlir::Type HiSPNDialect::parseType(::mlir::DialectAsmParser& parser) const {
  return ProbabilityType::get(getContext());
}

void HiSPNDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter& os) const {
  // Currently the only SPN type is the probability type.
  os << "probability";
}

// Add definitions/implementation of SPN dialect/operation interfaces.
#include "HiSPN/HiSPNInterfaces.cpp.inc"