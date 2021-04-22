//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPSeeding.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

SeedAnalysis::SeedAnalysis(Operation* rootOp) : rootOp{rootOp} {}

// Helper functions in anonymous namespace.
namespace {
  DenseMap<Value, unsigned> getOpDepths(Operation* rootOp) {
    DenseMap<Value, unsigned> opDepths;
    rootOp->walk([&](Operation* op) {
      unsigned depth = 0;
      for (auto const& value : op->getOperands()) {
        if (!opDepths.count(value)) {
          opDepths[value] = 0;
        } else {
          depth = std::max(depth, opDepths[value] + 1);
        }
      }
      for (auto const& result : op->getResults()) {
        opDepths[result] = depth;
      }
    });
    return opDepths;
  }
}

vector_t SeedAnalysis::getSeed(unsigned width, SearchMode const& mode) const {

  llvm::StringMap<SmallVector<vector_t>> seedsByOpName;
  auto const& opDepths = getOpDepths(rootOp);

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
    for (auto& seed : seedsByOpName[op->getName().getStringRef()]) {
      if (seed.size() < width && opDepths.lookup(seed.front()) == depth) {
        // Cannot use values for seeds that are defined in different scopes.
        if (seed.front().getParentRegion() != value.getParentRegion()) {
          continue;
        }
        seed.emplace_back(value);
        needsNewSeed = false;
        break;
      }
    }
    if (needsNewSeed) {
      vector_t seed{value};
      seedsByOpName[op->getName().getStringRef()].emplace_back(seed);
    }
  });

  SmallVector<vector_t> seeds;
  // Flatten the map that maps opcode to seeds.
  for (auto const& entry : seedsByOpName) {
    for (auto const& seed : entry.second) {
      if (seed.size() != width) {
        continue;
      }
      seeds.emplace_back(seed);
    }
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
  return seeds.front();

}
