//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/Seeding.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

// Helper functions in anonymous namespace.
namespace {
  DenseMap<Value, unsigned> computeOpDepths(Operation* rootOp, std::unordered_set<Operation*> const& availableOps) {
    DenseMap<Value, unsigned> opDepths;
    rootOp->walk([&](Operation* op) {
      if (!availableOps.count(op)) {
        return WalkResult::advance();
      }
      unsigned depth = 0;
      for (auto const& operand : op->getOperands()) {
        if (!opDepths.count(operand)) {
          opDepths[operand] = 0;
        }
        depth = std::max(depth, opDepths[operand] + 1);
      }
      for (auto const& result : op->getResults()) {
        opDepths[result] = depth;
      }
      return WalkResult::advance();
    });
    return opDepths;
  }

  void computeAvailableOps(Operation* rootOp, std::unordered_set<Operation*>& availableOps) {
    rootOp->walk([&](Operation* op) {
      if (!vectorizable(op)) {
        return;
      }
      availableOps.emplace(op);
    });
  }
}

SeedAnalysis::SeedAnalysis(Operation* rootOp, unsigned width) : rootOp{rootOp}, width{width} {
  computeAvailableOps(rootOp, availableOps);
}

SmallVector<Value, 4> SeedAnalysis::next(Order const& mode) {
  auto opDepths = computeOpDepths(rootOp, availableOps);
  llvm::StringMap<SmallVector<SmallVector<Value, 4>>> seedsByOpName;
  rootOp->emitRemark("Computing seed out of " + std::to_string(availableOps.size()) + " operations...");
  SmallVector<Value, 4> seed;
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
    return seed;
  }

  // Sort the seeds such that either the seeds closest to the beginning of the function come first (DefUse),
  // or those closest to the return statement (UseDef).
  std::sort(seeds.begin(), seeds.end(), [&](auto const& seed1, auto const& seed2) {
    switch (mode) {
      case DefUse: return opDepths.lookup(seed1.front()) < opDepths.lookup(seed2.front());
      case UseDef: return opDepths.lookup(seed1.front()) > opDepths.lookup(seed2.front());
        // Unused so far.
      default: assert(false);
    }
  });
  return seeds.front();
}

void SeedAnalysis::markAllUnavailable(ValueVector* root) {
  for (auto* vector : graph::postOrder(root)) {
    for (auto const& element : *vector) {
      if (auto* definingOp = element.getDefiningOp()) {
        availableOps.erase(definingOp);
      }
    }
  }
}
