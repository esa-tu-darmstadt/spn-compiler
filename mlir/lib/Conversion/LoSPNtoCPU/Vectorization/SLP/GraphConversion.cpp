//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/GraphConversion.h"
#include "llvm/Support/Debug.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

// Helper functions in anonymous namespace.
namespace {

  std::pair<Operation*, bool> resolveInBetweenOperations(Operation* latestInput,
                                                         SmallPtrSetImpl<Operation*>& inputs,
                                                         Operation* earliestEscapingUser,
                                                         SmallPtrSetImpl<Operation*>& users) {
    // false = only serves as input/use, true = serves as both input and use
    DenseMap<Operation*, bool> isOnUseInputPath;
    // Determine inputs to move before the earliest escaping user.
    SmallVector<Operation*> inputsToMove;
    for (auto* input : inputs) {
      SmallVector<Operation*, 16> worklist{input};
      unsigned oldInputSize = inputsToMove.size();
      while (!worklist.empty()) {
        auto* currentOp = worklist.pop_back_val();
        if (currentOp->isBeforeInBlock(earliestEscapingUser)) {
          isOnUseInputPath[currentOp] = false;
          continue;
        }
        if (users.contains(currentOp)) {
          isOnUseInputPath[currentOp] = true;
        }
        if (isOnUseInputPath.lookup(currentOp)) {
          for (auto* user : currentOp->getUsers()) {
            if (user->isBeforeInBlock(latestInput)) {
              isOnUseInputPath[user] = true;
              worklist.emplace_back(user);
            }
          }
          continue;
        }
        inputsToMove.emplace_back(currentOp);
        for (auto const& operand : currentOp->getOperands()) {
          if (auto* operandOp = operand.getDefiningOp()) {
            worklist.emplace_back(operandOp);
          }
        }
      }
      if (isOnUseInputPath.lookup(input)) {
        inputsToMove.pop_back_n(inputsToMove.size() - oldInputSize);
      }
    }
    if (inputsToMove.empty()) {
      return std::make_pair(earliestEscapingUser, true);
    }
    std::sort(std::begin(inputsToMove), std::end(inputsToMove), [&](Operation* lhs, Operation* rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    inputsToMove.erase(std::unique(std::begin(inputsToMove), std::end(inputsToMove)), std::end(inputsToMove));

    // Determine users to move after the latest input.
    SmallVector<Operation*> usersToMove;
    for (auto* user : users) {
      if (isOnUseInputPath.lookup(user)) {
        continue;
      }
      SmallVector<Operation*, 16> worklist{user};
      SmallPtrSet<Operation*, 32> seenUsers;
      while (!worklist.empty()) {
        auto* currentOp = worklist.pop_back_val();
        seenUsers.insert(currentOp);
        if (latestInput->isBeforeInBlock(currentOp)) {
          continue;
        }
        usersToMove.emplace_back(currentOp);
        for (auto* nextUser : currentOp->getUsers()) {
          if (!seenUsers.contains(nextUser)) {
            worklist.emplace_back(nextUser);
          }
        }
      }
    }
    if (usersToMove.empty()) {
      return std::make_pair(latestInput, false);
    }
    std::sort(std::begin(usersToMove), std::end(usersToMove), [&](Operation* lhs, Operation* rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    usersToMove.erase(std::unique(std::begin(usersToMove), std::end(usersToMove)), std::end(usersToMove));

    Operation* insertionPoint;

    llvm::dbgs() << "inputs to move:\n";
    for (auto* input : inputsToMove) {
      llvm::dbgs() << "\t" << *input << "\n";
    }
    llvm::dbgs() << "users to move:\n";
    for (auto* user : usersToMove) {
      llvm::dbgs() << "\t" << *user << "\n";
    }

    if (inputsToMove.size() < usersToMove.size()) {
      llvm::dbgs() << "moving inputs...\n";
      for (auto* input : inputsToMove) {
        input->moveBefore(earliestEscapingUser);
      }
      insertionPoint = inputsToMove.back();
    } else {
      llvm::dbgs() << "moving users...\n";
      for (size_t i = 0; i < usersToMove.size(); ++i) {
        usersToMove[i]->moveAfter(i == 0 ? latestInput : usersToMove[i - 1]);
      }
      insertionPoint = latestInput;
    }

    return std::make_pair(insertionPoint, false);

  }

}

ConversionManager::ConversionManager(ArrayRef<SLPNode const*> const& nodes) {
  Operation* latestInput = nullptr;
  SmallPtrSet<Operation*, 32> inputs;
  for (auto const* node : nodes) {
    for (size_t i = node->numVectors(); i-- > 0;) {
      auto* vector = node->getVector(i);
      for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
        auto const& element = vector->getElement(lane);
        if (auto* elementOp = element.getDefiningOp()) {
          if (!escapingUsers.count(element)) {
            escapingUsers[element].assign(std::begin(element.getUsers()), std::end(element.getUsers()));
          }
          for (auto const& operand : elementOp->getOperands()) {
            auto& users = escapingUsers[operand];
            users.erase(std::remove(std::begin(users), std::end(users), elementOp), std::end(users));
          }
          if (vector->numOperands() == 0) {
            dumpSLPNodeVector(*vector);
            for (size_t j = 0; j < vector->numLanes(); ++j) {
              auto const& copy = vector->getElement(j);
              if (auto* copyOp = copy.getDefiningOp()) {
                for (auto const& oper : copyOp->getOperands()) {
                  llvm::dbgs() << "\t" << oper << " (" << (oper.getDefiningOp() ? oper.getDefiningOp() : 0) << ")\n";
                }
              }
            }
            if (!latestInput || latestInput->isBeforeInBlock(elementOp)) {
              latestInput = elementOp;
            }
            inputs.insert(elementOp);
          }
        }
      }
    }
  }
  if (!latestInput) {
    // No non-block-argument-input => SLP graph insertion point can be at the beginning of the basic block (i.e. null).
    return;
  }
  llvm::dbgs() << "latest input: " << *latestInput << "\n";
  for (auto* input : inputs) {
    llvm::dbgs() << "\t" << *input << "\n";
  }

  Operation* earliestEscapingUser = nullptr;
  SmallPtrSet<Operation*, 32> users;
  for (auto& entry : escapingUsers) {
    // Sort already for later uses.
    std::sort(std::begin(entry.second), std::end(entry.second), [&](Operation* lhs, Operation* rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    for (auto* escapingUser : entry.second) {
      if (!earliestEscapingUser || escapingUser->isBeforeInBlock(earliestEscapingUser)) {
        earliestEscapingUser = escapingUser;
      }
      users.insert(escapingUser);
    }
  }
  if (users.empty()) {
    insertionPoint = std::make_pair(latestInput, false);
    return;
  }
  llvm::dbgs() << "earliest escaping user: " << *earliestEscapingUser << "\n";
  for (auto* user : users) {
    llvm::dbgs() << "\t" << *user << "\n";
  }
  insertionPoint = resolveInBetweenOperations(latestInput, inputs, earliestEscapingUser, users);
  llvm::dbgs() << "insertion point (" << (insertionPoint->second ? "before" : "after") << " this one): "
               << *insertionPoint->first << "\n";
}

void ConversionManager::setInsertionPointFor(NodeVector* vector, PatternRewriter& rewriter) const {
  Operation* latestOp = nullptr;
  for (size_t i = 0; i < vector->numOperands(); ++i) {
    auto* operand = vector->getOperand(i);
    assert(vectorData.lookup(operand).operation.hasValue() && "operand has not yet been converted");
    auto* operandOp = vectorData.lookup(operand).operation->getDefiningOp();
    if (!latestOp || latestOp->isBeforeInBlock(operandOp)) {
      latestOp = operandOp;
    }
  }
  if (latestOp) {
    rewriter.setInsertionPointAfter(latestOp);
  } else if (insertionPoint.hasValue()) {
    if (insertionPoint->second) {
      rewriter.setInsertionPoint(insertionPoint->first);
    } else {
      rewriter.setInsertionPointAfter(insertionPoint->first);
    }
  } else {
    rewriter.setInsertionPointToStart(vector->getElement(0).getParentBlock());
  }
}

void ConversionManager::update(NodeVector* vector, Value const& operation, ElementFlag const& flag) {
  assert(!vectorData[vector].operation.hasValue() && !vectorData[vector].flag.hasValue()
             && "vector has been converted already");
  vectorData[vector].operation = operation;
  vectorData[vector].flag = flag;
  if (insertionPoint->first->isBeforeInBlock(operation.getDefiningOp())) {
    insertionPoint = std::make_pair(operation.getDefiningOp(), false);
  }
}

bool ConversionManager::wasConverted(NodeVector* vector) const {
  return vectorData.lookup(vector).operation.hasValue();
}

Value ConversionManager::getValue(NodeVector* vector) const {
  assert(wasConverted(vector) && "vector has not yet been converted");
  return vectorData.lookup(vector).operation.getValue();
}

ElementFlag ConversionManager::getElementFlag(NodeVector* vector) const {
  assert(vectorData.lookup(vector).flag.hasValue() && "vector has not yet been converted");
  return vectorData.lookup(vector).flag.getValue();
}

bool ConversionManager::hasEscapingUsers(Value const& value) const {
  return escapingUsers.count(value) && !escapingUsers.lookup(value).empty();
}

Operation* ConversionManager::getEarliestEscapingUser(Value const& value) const {
  assert(hasEscapingUsers(value) && "value does not have any escaping users");
  return escapingUsers.lookup(value).front();
}
