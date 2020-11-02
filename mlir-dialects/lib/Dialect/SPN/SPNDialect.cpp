//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/SPNDialect.h"
#include "SPN/SPNOps.h"
#include "SPN/SPNAttributes.h"
#include "SPN/SPNInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include <vector>

using namespace mlir;
using namespace mlir::spn;

void SPNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SPN/SPNOps.cpp.inc"
  >();
  addTypes<ProbabilityType>();
}

::mlir::Type SPNDialect::parseType(::mlir::DialectAsmParser& parser) const {
  return ProbabilityType::get(getContext());
}

void SPNDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter& os) const {
  // Currently the only SPN type is the probability type.
  os << "probability";
}

// Add definitions/implementation of SPN dialect/operation interfaces.
#include "SPN/SPNInterfaces.cpp.inc"