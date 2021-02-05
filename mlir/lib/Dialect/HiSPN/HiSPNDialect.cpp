//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "HiSPN/HiSPNDialect.h"
#include "HiSPN/HiSPNOps.h"
#include "HiSPN/HiSPNAttributes.h"
#include "HiSPN/HiSPNInterfaces.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::spn::high;

void HiSPNDialect::initialize() {
  addOperations<
#define GET_OP_LIST \
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