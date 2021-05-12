//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/CostModel.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

// === CostModel === //

unsigned CostModel::getScalarCost(Value const& value) {
  auto const& entry = cachedScalarCosts.try_emplace(value, 0);
  if (entry.second) {
    entry.first->getSecond() = computeScalarCost(value);
  }
  return entry.first->second;
}

unsigned CostModel::getVectorCost(ValueVector const* vector) {
  auto const& entry = cachedVectorCosts.try_emplace(vector, 0);
  if (entry.second) {
    entry.first->getSecond() = computeVectorCost(vector);
  }
  return entry.first->second;
}

// === UnitCostModel === //

unsigned UnitCostModel::computeScalarCost(Value const& value) {
  unsigned cost = 1;
  if (auto* definingOp = value.getDefiningOp()) {
    for (auto const& operand : definingOp->getOperands()) {
      cost += getScalarCost(operand);
    }
  }
  return cost;
}

unsigned UnitCostModel::computeVectorCost(ValueVector const* vector) {
  unsigned cost = 1;
  for (size_t i = 0; i < vector->numOperands(); ++i) {
    cost += getVectorCost(vector->getOperand(i));
  }
  return cost;
}
