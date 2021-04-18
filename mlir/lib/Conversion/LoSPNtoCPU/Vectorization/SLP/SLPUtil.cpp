//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"

using namespace mlir;

bool mlir::spn::low::slp::areConsecutiveLoads(low::SPNBatchRead load1, low::SPNBatchRead load2) {
  if (load1 == load2) {
    return false;
  }
  if (load1.batchMem() != load2.batchMem()) {
    return false;
  }
  if (load1.batchIndex() != load2.batchIndex()) {
    return false;
  }
  return load1.sampleIndex() + 1 == load2.sampleIndex();
}

bool mlir::spn::low::slp::areConsecutiveLoads(std::vector<Operation*> const& loads) {
  for (size_t i = 0; i < loads.size() - 1; ++i) {
    auto loadOp1 = dyn_cast<low::SPNBatchRead>(loads[i]);
    if (!loadOp1) {
      return false;
    }
    auto loadOp2 = dyn_cast<low::SPNBatchRead>(loads[i + 1]);
    if (!loadOp2) {
      return false;
    }
    if (!areConsecutiveLoads(loadOp1, loadOp2)) {
      return false;
    }
  }
  return true;
}
