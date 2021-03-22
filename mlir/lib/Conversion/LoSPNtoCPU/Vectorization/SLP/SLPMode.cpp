//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPMode.h"

using namespace mlir;
using namespace mlir::spn;

Mode mlir::spn::modeFromOperation(const Operation* operation) {
  if (dyn_cast<ConstantOp>(operation)) {
    return CONST;
  } else if (dyn_cast<LoadOp>(operation)) {
    return LOAD;
  }
  return OPCODE;
}

