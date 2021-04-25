//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPConversion.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

// Helper functions in anonymous namespace.
namespace {
  bool isBeforeInBlock(Value const& lhs, Value const& rhs) {
    if (lhs.isa<BlockArgument>()) {
      return !rhs.isa<BlockArgument>();
    } else if (rhs.isa<BlockArgument>()) {
      return false;
    }
    return lhs.getDefiningOp()->isBeforeInBlock(rhs.getDefiningOp());
  }

  Value firstUser(Value const& element) {
    Operation* first = nullptr;
    for (auto* user : element.getUsers()) {
      if (!first || user->isBeforeInBlock(first)) {
        first = user;
      }
    }
    return first->getResult(0);
  }

}

ConversionState::ConversionState(SLPNode* root) {
  DenseMap<Value, unsigned> outsideUses;
  for (auto* node : SLPNode::postOrder(root)) {
    for (size_t i = node->numVectors(); i-- > 0;) {
      auto* vector = node->getVector(i);
      for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
        auto const& element = vector->getElement(lane);
        // Store the graph's last input to determine the earliest possible insertion point.
        if (!earliestInsertionPoint || (vector->isLeaf() && isBeforeInBlock(earliestInsertionPoint, element))) {
          earliestInsertionPoint = element;
        }
        // Skip duplicate (splat) values.
        if (outsideUses.count(element)) {
          continue;
        }
        outsideUses[element] = std::distance(std::begin(element.getUsers()), std::end(element.getUsers()));
        for (size_t j = 0; j < vector->numOperands(); ++j) {
          auto* operand = vector->getOperand(j);
          assert(outsideUses[operand->getElement(lane)] > 0);
          outsideUses[operand->getElement(lane)]--;
        }
      }
    }
  }

  for (auto* node : SLPNode::postOrder(root)) {
    for (size_t i = 0; i < node->numVectors(); ++i) {
      auto* vector = node->getVector(i);
      for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
        auto const& element = vector->getElement(lane);
        if (outsideUses[element] > 0) {
          vectorData[vector].firstEscapingUses.insert(std::make_pair(lane, firstUser(element)));
        }
      }
    }
  }
}

Value ConversionState::getInsertionPoint(NodeVector* vector) const {
  Value insertionPoint = earliestInsertionPoint;
  for (size_t i = 0; i < vector->numOperands(); ++i) {
    auto* operand = vector->getOperand(i);
    if (!vectorData.count(operand)) {
      continue;
    }
    assert(vectorData.lookup(operand).operation.hasValue() && "operand has not yet been converted");
    auto const& operandValue = vectorData.lookup(operand).operation.getValue();
    if (isBeforeInBlock(insertionPoint, operandValue)) {
      insertionPoint = operandValue;
    }
  }
  return insertionPoint;
}

void ConversionState::update(NodeVector* vector, Value const& operation, CreationMode const& mode) {
  auto& data = vectorData[vector];
  assert(!data.operation.hasValue() && !data.mode.hasValue() && "vector has been converted already");
  data.operation = operation;
  data.mode = mode;
  if (vector->isLeaf()) {
    if (isBeforeInBlock(earliestInsertionPoint, operation)) {
      earliestInsertionPoint = operation;
    }
  }
}

bool ConversionState::isConverted(NodeVector* vector) const {
  return vectorData.lookup(vector).operation.hasValue();
}

Value ConversionState::getValue(NodeVector* vector) const {
  assert(!isConverted(vector) && "vector has not yet been converted");
  return vectorData.lookup(vector).operation.getValue();
}

CreationMode ConversionState::getCreationMode(NodeVector* vector) const {
  assert(vectorData.lookup(vector).mode.hasValue() && "vector has not yet been converted");
  return vectorData.lookup(vector).mode.getValue();
}

Optional<Value> ConversionState::getFirstEscapingUse(NodeVector* vector, size_t lane) const {
  return vectorData.lookup(vector).firstEscapingUses.lookup(lane);
}
