//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
#include "LoSPN/LoSPNOps.h"

using namespace mlir;

bool mlir::spn::low::slp::areConsecutiveLoads(Operation* op1, Operation* op2) {

  if (op1 == op2) {
    return false;
  }

  auto loadOp1 = dyn_cast<low::SPNBatchRead>(op1);
  auto loadOp2 = dyn_cast<low::SPNBatchRead>(op2);

  if (!loadOp1 || !loadOp2) {
    return false;
  }

  if (loadOp1.batchMem() != loadOp2.batchMem()) {
    return false;
  }

  if (loadOp1.batchIndex() != loadOp2.batchIndex()) {
    return false;
  }

  return loadOp1.sampleIndex() + 1 == loadOp2.sampleIndex();
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

bool mlir::spn::low::slp::canBeGathered(std::vector<Operation*> const& loads) {
  auto firstLoad = dyn_cast<SPNBatchRead>(loads.front());
  if (!firstLoad) {
    return false;
  }
  for (size_t i = 1; i < loads.size(); ++i) {
    auto load = dyn_cast<SPNBatchRead>(loads[i]);
    if (!load) {
      return false;
    }
    if (firstLoad.batchMem() != load.batchMem()) {
      return false;
    }
    if (firstLoad.batchIndex() != load.batchIndex()) {
      return false;
    }
  }
  return true;
}

bool mlir::spn::low::slp::areBroadcastable(std::vector<Operation*> const& ops) {
  return std::all_of(std::begin(ops), std::end(ops), [&](mlir::Operation* op) {
    return op == ops.front();
  });
}

