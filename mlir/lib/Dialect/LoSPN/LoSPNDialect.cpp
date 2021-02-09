//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNAttributes.h"
#include "LoSPN/LoSPNInterfaces.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::spn::low;

void LoSPNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LoSPN/LoSPNOps.cpp.inc"
  >();
}

// Add definitions/implementation of SPN dialect/operation interfaces.
#include "LoSPN/LoSPNInterfaces.cpp.inc"