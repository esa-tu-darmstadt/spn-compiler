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

/// Individual cost taken from NodePatterns.cpp
double UnitCostModel::computeScalarCost(Value const& value) {
  auto* definingOp = value.getDefiningOp();
  if (!definingOp) {
    return 0;
  }
  if (dyn_cast<ConstantOp>(definingOp) || dyn_cast<SPNConstant>(definingOp)) {
    return 0;
  } else if (auto histogramOp = dyn_cast<SPNHistogramLeaf>(definingOp)) {
    if (histogramOp.supportMarginal()) {
      return 3;
    }
  } else if (auto categoricalOp = dyn_cast<SPNCategoricalLeaf>(definingOp)) {
    if (categoricalOp.supportMarginal()) {
      return 3;
    }
  } else if (auto gaussianOp = dyn_cast<SPNGaussianLeaf>(definingOp)) {
    if (gaussianOp.getResult().getType().isa<LogType>()) {
      if (gaussianOp.supportMarginal()) {
        return 6;
      }
      return 4;
    } else if (gaussianOp.supportMarginal()) {
      return 7;
    }
    return 5;
  } else if (auto addOp = dyn_cast<SPNAdd>(definingOp)) {
    if (addOp.getResult().getType().isa<LogType>()) {
      return 9;
    }
  }
  // Assume a default cost of 1, i.e. that there roughly is a 1:1 mapping of operations to actual assembly instructions.
  return 1;
}

double UnitCostModel::singleElementExtractionCost() {
  return 1;
}

void UnitCostModel::visit(BroadcastSuperword* pattern, Superword* superword) {
  this->cost = getScalarTreeCost(superword->getElement(0)) + 1;
}

void UnitCostModel::visit(BroadcastInsertSuperword* pattern, Superword* superword) {
  this->cost = static_cast<double>(superword->numLanes());
  for (auto const& element : *superword) {
    this->cost += getScalarTreeCost(element);
  }
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
