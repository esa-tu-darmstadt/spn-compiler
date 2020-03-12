//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

namespace mlir {
#include "src/codegen/mlir/dialects/spn/SPNOps.attr.h.inc"

#include "src/codegen/mlir/dialects/spn/SPNOps.attr.cpp.inc"
}