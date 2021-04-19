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
      if (!vectorizable(op)) {
        return;
      }
      auto const& value = op->getResult(0);
      if (!opDepths.count(value)) {
        opDepths[value] = 0;
      }
      for (auto const& use : op->getUses()) {
        if (!opDepths.count(use.get()) || opDepths[value] + 1 > opDepths[use.get()]) {
          opDepths[use.get()] = opDepths[value] + 1;
        }
      }
    });
    return opDepths;
  }
}

vector_t SeedAnalysis::getSeed(unsigned width, SearchMode const& mode) const {

  llvm::StringMap<SmallVector<vector_t>> seedsByOpName;
  auto const& opDepths = getOpDepths(rootOp);

  for (auto const& entry : opDepths) {
    auto const& value = entry.first;
    auto const& depth = entry.second;
    if (depth < log2(width)) {
      continue;
    }
    bool needsNewSeed = true;
    for (auto& seed : seedsByOpName[value.getDefiningOp()->getName().getStringRef()]) {
      if (seed.size() < width && opDepths.lookup(seed.front()) == depth) {
        seed.emplace_back(entry.first);
        needsNewSeed = false;
        break;
      }
    }
    if (needsNewSeed) {
      vector_t seed{value};
      seedsByOpName[value.getDefiningOp()->getName().getStringRef()].emplace_back(seed);
    }
  }

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
