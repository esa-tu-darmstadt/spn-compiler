//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPSeeding.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

SeedAnalysis::SeedAnalysis(Operation* rootOp, unsigned width) : rootOp{rootOp}, width{width} {}

// Helper functions in anonymous namespace.
namespace {
  DenseMap<Value, unsigned> getOpDepths(Operation* rootOp) {
    DenseMap<Value, unsigned> opDepths;
    rootOp->walk([&](Operation* op) {
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
    });
    return opDepths;
  }
}

void SeedAnalysis::fillSeed(SmallVectorImpl<Value>& seed, SearchMode const& mode) const {
  llvm::StringMap<SmallVector<SmallVector<Value, 4>>> seedsByOpName;
  auto const& opDepths = getOpDepths(rootOp);
  rootOp->emitRemark("Computing seed out of " + std::to_string(opDepths.size()) + " operations...");
  rootOp->walk([&](Operation* op) {
    if (!vectorizable(op)) {
      return WalkResult::advance();
    }
    auto value = op->getResult(0);
    auto const& depth = opDepths.lookup(value);
    if (depth < log2(width)) {
      return WalkResult::advance();
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
      SmallVector<Value, 4> seed{value};
      seedsByOpName[op->getName().getStringRef()].emplace_back(seed);
    }
    return WalkResult::advance();
  });

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
    return;
  }

  // Sort the seeds such that either the seeds closest to the beginning of the function come first (DefBeforeUse),
  // or those closest to the return statement (UseBeforeDef).
  std::sort(seeds.begin(), seeds.end(), [&](auto const& seed1, auto const& seed2) {
    if (mode == DefBeforeUse) {
      return opDepths.lookup(seed1.front()) < opDepths.lookup(seed2.front());
    } else if (mode == UseBeforeDef) {
      return opDepths.lookup(seed1.front()) > opDepths.lookup(seed2.front());
    } else {
      // Unused so far.
      assert(false);
    }
  });

  seed.swap(seeds.front());
}
