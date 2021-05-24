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

unsigned CostModel::getSuperwordCost(Superword const& superword) {
  auto const& entry = cachedSuperwordCosts.try_emplace(&superword, 0);
  if (entry.second) {
    entry.first->getSecond() = computeSuperwordCost(superword);
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

unsigned UnitCostModel::computeSuperwordCost(Superword const& superword) {
  unsigned cost = 1;
  for (size_t i = 0; i < superword.numOperands(); ++i) {
    cost += getSuperwordCost(*superword.getOperand(i));
  }
  return cost;
}
