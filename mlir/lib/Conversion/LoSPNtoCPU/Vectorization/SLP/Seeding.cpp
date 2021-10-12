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
      return {};
    }
  }
  auto seed = nextSeed();
  for (auto value : seed) {
    if (auto* definingOp = value.getDefiningOp()) {
      availableOps.remove(definingOp);
    }
  }
  return seed;
}

void SeedAnalysis::update(ArrayRef<Superword*> convertedSuperwords) {
  SmallPtrSet<Operation*, 32> convertedOps;
  for (auto* superword : convertedSuperwords) {
    for (auto element : *superword) {
      if (auto* definingOp = element.getDefiningOp()) {
        convertedOps.insert(definingOp);
      }
    }
  }
  // Construct anew and swap because erasing many ops one after one takes a lot of time in set vectors.
  llvm::SmallSetVector<Operation*, 32> newAvailableOps;
  for (auto* op : availableOps) {
    if (!convertedOps.contains(op)) {
      newAvailableOps.insert(op);
    }
  }
  availableOps.swap(newAvailableOps);
}

// === TopDownSeedAnalysis === //

// Helper functions in anonymous namespace.
namespace {

  template<typename OpIterator>
  DenseMap<Value, unsigned> depths(OpIterator begin, OpIterator end) {
    DenseMap<Value, unsigned> opDepths;
    llvm::SmallVector<Operation*, 32> worklist{begin, end};
    while (!worklist.empty()) {
      auto* op = worklist.pop_back_val();
      auto depth = opDepths[op->getResult(0)];
      for (auto operand : op->getOperands()) {
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
                                                          llvm::SmallSetVector<Operation*, 32> const& availableOps,
                                                          unsigned width) {
    llvm::StringMap<SmallVector<SmallVector<Value, 4>>> seedsByOpName;
    SmallVector<Value> sortedCandidates;
    for (auto& entry : opDepths) {
      sortedCandidates.emplace_back(entry.first);
    }
    // Required for deterministic seed retrieval.
    // Sort s.t. operations that appear later in the program come first.
    llvm::sort(std::begin(sortedCandidates), std::end(sortedCandidates), [&](Value lhs, Value rhs) {
      if (auto* definingLhs = lhs.getDefiningOp()) {
        if (auto* definingRhs = rhs.getDefiningOp()) {
          return definingRhs->isBeforeInBlock(definingLhs);
        } else {
          return true;
        }
      }
      return false;
    });
    for (auto value : sortedCandidates) {
      auto* definingOp = value.getDefiningOp();
      if (!definingOp || definingOp->hasTrait<OpTrait::ConstantLike>() || !vectorizable(definingOp)
          || !availableOps.contains(definingOp)) {
        continue;
      }
      auto const& depth = opDepths.lookup(value);
      auto const& opName = definingOp->getName().getStringRef();
      if (depth < log2(width)) {
        continue;
      }
      bool needsNewSeed = true;
      for (auto& potentialSeed : seedsByOpName[opName]) {
        if (potentialSeed.size() < width && opDepths.lookup(potentialSeed.front()) == depth) {
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
    // Required for deterministic seed retrieval.
    // Sort s.t. seeds whose first operation appears later in the program come first.
    llvm::sort(std::begin(seeds), std::end(seeds), [&](auto const& lhs, auto const& rhs) {
      return rhs.front().getDefiningOp()->isBeforeInBlock(lhs.front().getDefiningOp());
    });
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
  auto opDepths = depths(std::begin(availableOps), std::end(availableOps));
  auto seeds = computeSeedsByOpName(opDepths, availableOps, width);

  if (seeds.empty()) {
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
  llvm::BitVector leafCoverage(DenseMap<Operation*, llvm::BitVector>& reachableLeaves, ArrayRef<Value> seed) {
    llvm::BitVector disjunction = reachableLeaves.lookup(seed.front().getDefiningOp());
    for (size_t i = 1; i < seed.size(); ++i) {
      disjunction |= reachableLeaves.lookup(seed[i].getDefiningOp());
    }
    return disjunction;
  }

  SmallPtrSet<Operation*, 32> getLeaves(Operation* rootOp) {
    SmallPtrSet<Operation*, 32> leaves;
    rootOp->walk([&](spn::low::LeafNodeInterface leaf) {
      leaves.insert(leaf);
    });
    return leaves;
  }
}

FirstRootAnalysis::FirstRootAnalysis(Operation* rootOp, unsigned width) : SeedAnalysis{rootOp, width} {}

SmallVector<Value, 4> FirstRootAnalysis::nextSeed() const {
  auto const& leaves = getLeaves(rootOp);
  llvm::StringMap<DenseMap<Operation*, llvm::BitVector>> reachableLeaves;

  auto* root = findRoot(leaves, reachableLeaves);
  auto opDepths = root ? depths(&root, &root + 1) : depths(std::begin(availableOps), std::end(availableOps));
  auto seeds = computeSeedsByOpName(opDepths, availableOps, width);

  if (seeds.empty()) {
    rootOp->emitRemark("No seed found.");
    return {};
  }

  SmallVector<Value, 4>* bestSeed = nullptr;
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
  rootOp->walk([&](Operation* op) {
    if (!vectorizable(op) || op->hasTrait<OpTrait::ConstantLike>()) {
      return;
    }
    availableOps.insert(op);
  });
}

Operation* FirstRootAnalysis::findRoot(SmallPtrSet<Operation*, 32> const& leaves,
                                       llvm::StringMap<DenseMap<Operation*, llvm::BitVector>>& reachableLeaves) const {
  SmallPtrSet<Operation*, 32> uniqueWorklist;
  std::queue<Operation*> worklist;
  unsigned index = 0;
  for (auto* leaf : leaves) {
    for (auto* user : leaf->getUsers()) {
      if (!availableOps.contains(user)) {
        continue;
      }
      auto it = reachableLeaves[user->getName().getStringRef()].try_emplace(user, leaves.size());
      auto& userReachable = it.first->second;
      userReachable.set(index);
      if (userReachable.all()) {
        return user;
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
      if (!availableOps.contains(user)) {
        continue;
      }
      auto it = reachableLeaves[user->getName().getStringRef()].try_emplace(user, leaves.size());
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
  return nullptr;
}