//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPN/LoSPNOps.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPMode.h"

using namespace mlir;

mlir::spn::low::slp::Mode mlir::spn::low::slp::modeFromValue(Value const& value) {
  if (value.isa<BlockArgument>()) {
    return SPLAT;
  }
  auto* definingOp = value.getDefiningOp();
  if (definingOp->hasTrait<OpTrait::ConstantLike>()) {
    return CONST;
  } else if (dyn_cast<mlir::spn::low::SPNBatchRead>(definingOp)) {
    return LOAD;
  }
  return OPCODE;
}

