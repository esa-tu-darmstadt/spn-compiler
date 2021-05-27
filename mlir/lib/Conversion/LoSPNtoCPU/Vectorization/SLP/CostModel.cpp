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

// === UnitCostModel === //

double UnitCostModel::computeScalarCost(Value const& value) {
  return 1;
}

void UnitCostModel::visit(VectorizeConstant* pattern, Superword* superword) {
  this->cost = 0;
}

void UnitCostModel::visit(VectorizeBatchRead* pattern, Superword* superword) {
  this->cost = 1;
}

void UnitCostModel::visit(VectorizeAdd* pattern, Superword* superword) {
  this->cost = 1;
}

void UnitCostModel::visit(VectorizeMul* pattern, Superword* superword) {
  this->cost = 1;
}

void UnitCostModel::visit(VectorizeGaussian* pattern, Superword* superword) {
  this->cost = 6;
}
