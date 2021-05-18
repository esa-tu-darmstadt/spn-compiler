//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

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

unsigned CostModel::getVectorCost(ValueVector const& vector) {
  auto const& entry = cachedVectorCosts.try_emplace(&vector, 0);
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

unsigned UnitCostModel::computeVectorCost(ValueVector const& vector) {
  unsigned cost = 1;
  for (size_t i = 0; i < vector.numOperands(); ++i) {
    cost += getVectorCost(*vector.getOperand(i));
  }
  return cost;
}
