//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/Seeding.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include <queue>

using namespace mlir;
using namespace mlir::spn::low::slp;


// === SeedAnalysis === //

SeedAnalysis::SeedAnalysis(Operation* rootOp, unsigned width) : rootOp{rootOp}, width{width} {}

SmallVector<Value, 4> SeedAnalysis::next() {
  if (exhausted) {
    return {};
  }
  if (availableOps.empty()) {
    computeAvailableOps();
  }
  auto seed = nextSeed();
  if (seed.empty()) {
    exhausted = true;
  }
  return seed;
}

void SeedAnalysis::markAllUnavailable(Superword* root) {
  for (auto* vector : graph::postOrder(root)) {
    for (auto const& element : *vector) {
      if (auto* definingOp = element.getDefiningOp()) {
        availableOps.erase(definingOp);
      }
    }
  }
}

// === TopDownSeedAnalysis === //

TopDownAnalysis::TopDownAnalysis(Operation* rootOp, unsigned width) : SeedAnalysis{rootOp, width} {}

void TopDownAnalysis::computeAvailableOps() {
  rootOp->walk([&](Operation* op) {
    if (!vectorizable(op)) {
      return;
    }
    availableOps.insert(op);
  });
}

SmallVector<Value, 4> TopDownAnalysis::nextSeed() const {
  auto opDepths = computeOpDepths();
  llvm::StringMap<SmallVector<SmallVector<Value, 4>>> seedsByOpName;
  rootOp->emitRemark("Computing seed out of " + std::to_string(availableOps.size()) + " operations...");
  for (auto* op : availableOps) {
    auto value = op->getResult(0);
    auto const& depth = opDepths.lookup(value);
    if (depth < log2(width)) {
      continue;
    }
    bool needsNewSeed = true;
    for (auto& potentialSeed : seedsByOpName[op->getName().getStringRef()]) {
      if (potentialSeed.size() < width && opDepths.lookup(potentialSeed.front()) == depth) {
        // Cannot use values for seeds that are defined in different scopes.
        if (potentialSeed.front().getParentRegion() != value.getParentRegion()) {
          continue;
        }
        potentialSeed.emplace_back(value);
        if (potentialSeed.size() == width && depth == log2(width)) {
          return potentialSeed;
        }
        needsNewSeed = false;
        break;
      }
    }
    if (needsNewSeed) {
      SmallVector<Value, 4> newSeed{value};
      seedsByOpName[op->getName().getStringRef()].emplace_back(newSeed);
    }
  }

  SmallVector<SmallVector<Value, 4>> seeds;
  // Flatten the map that maps opcodes to seeds.
  for (auto const& entry : seedsByOpName) {
    for (auto const& potentialSeed : entry.second) {
      if (potentialSeed.size() != width) {
        continue;
      }
      seeds.emplace_back(potentialSeed);
    }
  }

  if (seeds.empty()) {
    rootOp->emitRemark("No seed found.");
    return {};
  }

  SmallVector<Value, 4>* seed = nullptr;
  for (auto& potentialSeed : seeds) {
    if (!seed) {
      seed = &potentialSeed;
      continue;
    }
    if (opDepths.lookup(potentialSeed.front()) < opDepths.lookup(seed->front())) {
      seed = &potentialSeed;
    }
  }
  return *seed;
}

DenseMap<Value, unsigned> TopDownAnalysis::computeOpDepths() const {
  DenseMap<Value, unsigned> opDepths;
  llvm::SmallVector<Operation*, 32> worklist{std::begin(availableOps), std::end(availableOps)};
  while (!worklist.empty()) {
    auto* op = worklist.pop_back_val();
    auto depth = opDepths[op->getResult(0)];
    for (auto const& operand : op->getOperands()) {
      if (auto* definingOp = operand.getDefiningOp()) {
        auto& operandDepth = opDepths[operand];
        if (depth + 1 > operandDepth) {
          operandDepth = depth + 1;
          worklist.emplace_back(definingOp);
        }
      }
    }
  }
  return opDepths;
}

// === BottomUpSeedAnalysis === //

BottomUpAnalysis::BottomUpAnalysis(Operation* rootOp, unsigned width) : SeedAnalysis{rootOp, width} {}

SmallVector<Value, 4> BottomUpAnalysis::nextSeed() const {
  llvm::StringMap<DenseMap<Operation*, llvm::BitVector>> reachableLeaves;
  auto root = findFirstRoot(reachableLeaves);
  for (auto const& nameEntry : reachableLeaves) {
    llvm::dbgs() << nameEntry.first() << ":\n";
    for (auto const& entry : nameEntry.second) {
      llvm::dbgs() << "\t" << *entry.first << " (" << entry.first << "):\t";
      for (size_t i = 0; i < entry.second.size(); ++i) {
        llvm::dbgs() << (entry.second.test(i) ? "1" : "0");
      }
      llvm::dbgs() << "\n";
    }
  }
  root->dump();
  llvm::dbgs() << root << "\n";
  return {};
}

void BottomUpAnalysis::computeAvailableOps() {
  rootOp->walk([&](LeafNodeInterface leaf) {
    availableOps.insert(leaf);
  });
}

Operation* BottomUpAnalysis::findFirstRoot(llvm::StringMap<DenseMap<Operation*,
                                                                    llvm::BitVector>>& reachableLeaves) const {
  SmallPtrSet<Operation*, 32> uniqueWorklist;
  std::queue<Operation*> worklist;
  unsigned index = 0;
  for (auto* op : availableOps) {
    for (auto* user : op->getUsers()) {
      auto it = reachableLeaves[user->getName().getStringRef()].try_emplace(user, availableOps.size());
      auto& userReachable = it.first->second;
      userReachable.set(index);
      if (userReachable.all()) {
        return op;
      }
      if (uniqueWorklist.insert(user).second) {
        worklist.emplace(user);
      }
    }
    ++index;
  }
  while (!worklist.empty()) {
    auto* currentOp = worklist.front();
    auto const& currentName = currentOp->getName().getStringRef();
    for (auto* user : currentOp->getUsers()) {
      auto it = reachableLeaves[user->getName().getStringRef()].try_emplace(user, availableOps.size());
      bool isNewUser = it.second;
      auto& userReachable = it.first->second;
      if (!isNewUser) {
        if (userReachable == reachableLeaves[currentName][currentOp]) {
          continue;
        }
      }
      userReachable |= reachableLeaves[currentName][currentOp];
      if (userReachable.all()) {
        return user;
      }
      if (uniqueWorklist.insert(user).second) {
        worklist.emplace(user);
      }
    }
    worklist.pop();
    uniqueWorklist.erase(currentOp);
  }
  llvm_unreachable("block contains unreachable operations");
}