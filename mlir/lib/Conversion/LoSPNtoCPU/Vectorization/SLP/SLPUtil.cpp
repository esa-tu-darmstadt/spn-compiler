//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

bool mlir::spn::low::slp::areConsecutiveLoads(Operation* load1, Operation* load2) {

  if (load1 == load2) {
    return false;
  }

  auto loadOp1 = dyn_cast<LoadOp>(load1);
  auto loadOp2 = dyn_cast<LoadOp>(load2);

  if (!loadOp1 || !loadOp2) {
    return false;
  }

  if (loadOp1.indices().size() != loadOp2.indices().size()) {
    return false;
  }

  for (size_t index = 0; index < loadOp1.indices().size(); ++index) {
    auto constOp1 = loadOp1.indices()[index].getDefiningOp<ConstantOp>();
    auto constOp2 = loadOp2.indices()[index].getDefiningOp<ConstantOp>();
    if (!constOp1 || !constOp2) {
      return false;
    }
    auto indexVal1 = constOp1.value().dyn_cast<IntegerAttr>();
    auto indexVal2 = constOp2.value().dyn_cast<IntegerAttr>();
    if (!indexVal1 || !indexVal2) {
      return false;
    }
    if ((index == loadOp1.indices().size() - 1 && indexVal1.getInt() != indexVal2.getInt() - 1)
        || indexVal1.getInt() != indexVal2.getInt()) {
      return false;
    }
  }
  return true;
}

bool mlir::spn::low::slp::areConsecutiveLoads(std::vector<Operation*> const& loads) {
  for (size_t i = 0; i < loads.size() - 1; ++i) {
    auto* loadOp1 = loads[i];
    auto* loadOp2 = loads[i + 1];
    if (!areConsecutiveLoads(loadOp1, loadOp2)) {
      return false;
    }
  }
  return true;
}

