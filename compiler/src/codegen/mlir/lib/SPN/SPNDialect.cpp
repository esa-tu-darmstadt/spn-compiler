//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/SPNDialect.h"
#include "SPN/SPNOps.h"

using namespace mlir;
using namespace mlir::spn;

void SPNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SPN/SPNOps.cpp.inc"
  >();
}