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
  if (availableOps.empty()) {
    if (!availableComputed) {
      computeAvailableOps();
      availableComputed = true;
    } else {
      rootOp->emitRemark("No seed found.");
      return {};
    }
  }
  rootOp->emitRemark("Computing seed out of " + std::to_string(availableOps.size()) + " operations...");
  auto seed = nextSeed();
  for (auto const& value : seed) {
    if (auto* definingOp = value.getDefiningOp()) {
      availableOps.erase(definingOp);
    }
  }
  return seed;
}

void SeedAnalysis::update() {
  availableOps.clear();
  availableComputed = false;
}

// === TopDownSeedAnalysis === //

// Helper functions in anonymous namespace.
namespace {

  template<typename OpIterator>
  DenseMap<Value, unsigned> computeOpDepths(OpIterator begin, OpIterator end) {
    DenseMap<Value, unsigned> opDepths;
    llvm::SmallVector<Operation*, 32> worklist{begin, end};
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

  SmallVector<SmallVector<Value, 4>> computeSeedsByOpName(DenseMap<Value, unsigned int>& opDepths,
                                                          SmallPtrSetImpl<Operation*> const& availableOps,
                                                          unsigned width) {
    llvm::StringMap<SmallVector<SmallVector<Value, 4>>> seedsByOpName;
    for (auto& entry : opDepths) {
      auto& value = entry.first;
      auto* definingOp = value.getDefiningOp();
      if (!definingOp || definingOp->hasTrait<OpTrait::ConstantLike>() || !vectorizable(definingOp)
          || !availableOps.contains(definingOp)) {
        continue;
      }
      auto const& depth = entry.second;
      auto const& opName = definingOp->getName().getStringRef();
      if (depth < log2(width)) {
        continue;
      }
      bool needsNewSeed = true;
      for (auto& potentialSeed : seedsByOpName[opName]) {
        if (potentialSeed.size() < width && opDepths.lookup(potentialSeed.front()) == depth) {
          //if (potentialSeed.size() < width && (opDepths.lookup(potentialSeed.front()) == 3 || opDepths.lookup(potentialSeed.front()) == 4)) {
          // Cannot use values for seeds that are defined in different blocks.
          if (potentialSeed.front().getParentBlock() != value.getParentBlock()) {
            continue;
          }
          potentialSeed.emplace_back(value);
          needsNewSeed = false;
          break;
        }
      }
      if (needsNewSeed) {
        SmallVector<Value, 4> newSeed{value};
        seedsByOpName[opName].emplace_back(newSeed);
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
    return seeds;
  }
}

TopDownAnalysis::TopDownAnalysis(Operation* rootOp, unsigned width) : SeedAnalysis{rootOp, width} {}

void TopDownAnalysis::computeAvailableOps() {
  rootOp->walk([&](Operation* op) {
    if (!vectorizable(op) || op->hasTrait<OpTrait::ConstantLike>()) {
      return;
    }
    availableOps.insert(op);
  });
}

SmallVector<Value, 4> TopDownAnalysis::nextSeed() const {
  auto opDepths = computeOpDepths(std::begin(availableOps), std::end(availableOps));
  auto seeds = computeSeedsByOpName(opDepths, availableOps, width);

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

// === FirstRootAnalysis === //

// Helper functions in anonymous namespace.
namespace {
  llvm::BitVector leafCoverage(DenseMap<Operation*, llvm::BitVector>& reachableLeaves, ArrayRef<Value> const& seed) {
    llvm::BitVector disjunction = reachableLeaves.lookup(seed.front().getDefiningOp());
    for (size_t i = 1; i < seed.size(); ++i) {
      disjunction |= reachableLeaves.lookup(seed[i].getDefiningOp());
    }
    return disjunction;
  }
}

FirstRootAnalysis::FirstRootAnalysis(Operation* rootOp, unsigned width) : SeedAnalysis{rootOp, width} {}

SmallVector<Value, 4> FirstRootAnalysis::nextSeed() const {
  llvm::StringMap<DenseMap<Operation*, llvm::BitVector>> reachableLeaves;
  auto* root = findFirstRoot(reachableLeaves);

  for (auto const& nameEntry : reachableLeaves) {
    llvm::dbgs() << nameEntry.first() << ":\n";
    for (auto const& entry : nameEntry.second) {
      llvm::dbgs() << "\t" << entry.first << ":\t";
      for (size_t i = 0; i < entry.second.size(); ++i) {
        llvm::dbgs() << (entry.second.test(i) ? "1" : "0");
      }
      llvm::dbgs() << "\n";
    }
  }

  auto opDepths = computeOpDepths(&root, &root + 1);
  // TODO: available ops computation
  auto seeds = computeSeedsByOpName(opDepths, availableOps, width);

  if (seeds.empty()) {
    rootOp->emitRemark("No seed found.");
    return {};
  }
  SmallVector<Value, 4>* bestSeed;
  unsigned bestLeafCoverage = 0;
  for (size_t i = 0; i < seeds.size(); ++i) {
    auto const& opName = seeds[i].front().getDefiningOp()->getName().getStringRef();
    auto leavesCovered = leafCoverage(reachableLeaves[opName], seeds[i]);
    if (!bestSeed || leavesCovered.count() > bestLeafCoverage) {
      bestSeed = &seeds[i];
      if (leavesCovered.all()) {
        break;
      }
      bestLeafCoverage = leavesCovered.count();
    }
  }
  return *bestSeed;
}

void FirstRootAnalysis::computeAvailableOps() {
  rootOp->walk([&](LeafNodeInterface leaf) {
    if (!vectorizable(leaf)) {
      return;
    }
    availableOps.insert(leaf);
  });
}

Operation* FirstRootAnalysis::findFirstRoot(llvm::StringMap<DenseMap<Operation*,
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