//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPMode.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

Mode slp::modeFromOperation(Operation const* operation) {
  if (dyn_cast<low::SPNConstant>(operation)) {
    return CONST;
  }
  // We don't have LOADs. Therefore just return OPCODE.
  return OPCODE;
}

