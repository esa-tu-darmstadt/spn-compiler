//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/CostModel.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

// === CostModel === //

CostModel::CostModel(SLPPatternApplicator const& applicator) : cost{0}, patternApplicator{applicator} {}

double CostModel::getScalarCost(Value value) {
  if (conversionState->alreadyComputed(value)) {
    return 0;
  }
  auto it = cachedScalarCost.try_emplace(value, computeScalarCost(value));
  if (it.second) {
    if (auto* definingOp = value.getDefiningOp()) {
      for (auto operand : definingOp->getOperands()) {
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
      auto* operandPattern = patternApplicator.bestMatch(operand);
      operandPattern->accept(*this, operand);
      vectorCost += cost;
    }
  }
  return vectorCost;
}

bool CostModel::isExtractionProfitable(Value value) {
  auto extractionCost = getExtractionCost(value);
  if (extractionCost == MAX_COST) {
    return false;
  }
  auto scalarCost = getScalarCost(value);
  return extractionCost < scalarCost;
}

void CostModel::setConversionState(std::shared_ptr<ConversionState> newConversionState) {
  conversionState = std::move(newConversionState);
  conversionState->addScalarCallbacks(
      [&](Value value) {
        updateCost(value, 0, false);
      }, [&](Value value) {
        cachedScalarCost.erase(value);
      }
  );
  conversionState->addExtractionCallbacks(
      [&](Value value) {
        updateCost(value, 0, true);
      }, [&](Value value) {
        cachedScalarCost.erase(value);
      }
  );
}

double CostModel::getBlockCost(Block* block, SmallPtrSetImpl<Operation*> const& deadOps) const {
  double blockCost = 0;
  block->walk([&](Operation* op) {
    if (deadOps.contains(op)) {
      return WalkResult::skip();
    }
    for (auto const& result : op->getResults()) {
      blockCost += computeScalarCost(result);
      // Assume that all results are computed at the same time.
      break;
    }
    return WalkResult::advance();
  });
  return blockCost;
}

void CostModel::updateCost(Value value, double newCost, bool updateUses) {
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

double CostModel::getExtractionCost(Value value) const {
  if (conversionState->isExtractable(value)) {
    auto valuePosition = conversionState->getSuperwordContainingValue(value);
    return computeExtractionCost(valuePosition.superword, valuePosition.index);
  }
  return MAX_COST;
}

// === UnitCostModel === //

/// Individual cost taken from NodePatterns.cpp
double UnitCostModel::computeScalarCost(Value value) const {
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

double UnitCostModel::computeExtractionCost(Superword* superword, size_t index) const {
  return 1;
}

void UnitCostModel::visitDefault(SLPVectorizationPattern const* pattern, Superword* superword) {
  this->cost = 1;
}

void UnitCostModel::visit(BroadcastSuperword const* pattern, Superword* superword) {
  this->cost = 1;
  for (auto element : leafVisitor.getRequiredScalarValues(pattern, superword)) {
    this->cost += getScalarCost(element);
  }
}

void UnitCostModel::visit(BroadcastInsertSuperword const* pattern, Superword* superword) {
  this->cost = 0;
  for (auto element : leafVisitor.getRequiredScalarValues(pattern, superword)) {
    this->cost += getScalarCost(element) + 1;
  }
}

void UnitCostModel::visit(VectorizeConstant const* pattern, Superword* superword) {
  this->cost = 0;
}

void UnitCostModel::visit(VectorizeSPNConstant const* pattern, Superword* superword) {
  this->cost = 0;
}

void UnitCostModel::visit(VectorizeGaussian const* pattern, Superword* superword) {
  if (anyGaussianMarginalized(*superword)) {
    this->cost = 7;
  }
  this->cost = 5;
}

void UnitCostModel::visit(VectorizeLogAdd const* pattern, Superword* superword) {
  this->cost = 4;
}

void UnitCostModel::visit(VectorizeLogGaussian const* pattern, Superword* superword) {
  if (anyGaussianMarginalized(*superword)) {
    this->cost = 6;
  }
  this->cost = 4;
}
