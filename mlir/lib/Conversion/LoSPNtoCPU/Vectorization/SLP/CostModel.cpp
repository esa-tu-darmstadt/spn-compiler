//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/CostModel.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "LoSPN/LoSPNTypes.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

// === CostModel === //

double CostModel::getScalarCost(Value const& value) {
  if (conversionState->alreadyComputed(value)) {
    return 0;
  }
  auto it = cachedScalarCost.try_emplace(value, computeScalarCost(value));
  if (it.second) {
    if (auto* definingOp = value.getDefiningOp()) {
      for (auto const& operand : definingOp->getOperands()) {
        if (isExtractionProfitable(operand)) {
          cachedScalarCost[value] += getExtractionCost(operand);
        } else {
          cachedScalarCost[value] += getScalarCost(operand);
        }
      }
    }
    return cachedScalarCost[value];
  }
  return it.first->second;
}

double CostModel::getSuperwordCost(Superword* superword, SLPVectorizationPattern* pattern) {
  if (conversionState->alreadyComputed(superword)) {
    return 0;
  }
  pattern->accept(*this, superword);
  double vectorCost = cost;
  for (auto* operand: superword->getOperands()) {
    if (!conversionState->alreadyComputed(superword)) {
      pattern->accept(*this, operand);
      vectorCost += cost;
    }
  }
  return vectorCost;
}

bool CostModel::isExtractionProfitable(Value const& value) {
  auto extractionCost = getExtractionCost(value);
  if (extractionCost == MAX_COST) {
    return false;
  }
  auto scalarCost = getScalarCost(value);
  return extractionCost < scalarCost;
}

void CostModel::setConversionState(std::shared_ptr<ConversionState> const& newConversionState) {
  conversionState = newConversionState;
  conversionState->addScalarCallback([&](Value value) {
    updateCost(value, 0, false);
  });
  conversionState->addExtractionCallback([&](Value value) {
    updateCost(value, 0, true);
  });
}

void CostModel::updateCost(Value const& value, double newCost, bool updateUses) {
  auto it = cachedScalarCost.try_emplace(value, newCost);
  if (!it.second && it.first->second <= newCost) {
    return;
  }
  if (updateUses) {
    for (auto* user : value.getUsers()) {
      for (auto const& result : user->getResults()) {
        if (cachedScalarCost.count(result)) {
          updateCost(result, getScalarCost(result), true);
        }
      }
    }
  }
}

double CostModel::getExtractionCost(Value const& value) {
  auto valuePosition = conversionState->getWordContainingValue(value);
  if (valuePosition.superword) {
    return computeExtractionCost(valuePosition.superword, valuePosition.index);
  }
  return MAX_COST;
}

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

double UnitCostModel::computeExtractionCost(Superword* superword, size_t index) {
  return 1;
}

void UnitCostModel::visitDefault(SLPVectorizationPattern* pattern, Superword* superword) {
  this->cost = 1;
}

void UnitCostModel::visit(BroadcastSuperword* pattern, Superword* superword) {
  this->cost = 1;
  for (auto const& element : scalarVisitor.getRequiredScalarValues(pattern, superword)) {
    this->cost += getScalarCost(element);
  }
}

void UnitCostModel::visit(BroadcastInsertSuperword* pattern, Superword* superword) {
  this->cost = 0;
  for (auto const& element : scalarVisitor.getRequiredScalarValues(pattern, superword)) {
    this->cost += getScalarCost(element) + 1;
  }
}

void UnitCostModel::visit(VectorizeConstant* pattern, Superword* superword) {
  this->cost = 0;
}

void UnitCostModel::visit(VectorizeGaussian* pattern, Superword* superword) {
  this->cost = 6;
}

void UnitCostModel::visit(VectorizeLogConstant* pattern, Superword* superword) {
  this->cost = 0;
}
