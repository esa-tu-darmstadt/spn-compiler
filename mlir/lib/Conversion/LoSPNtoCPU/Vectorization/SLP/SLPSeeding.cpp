//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPSeeding.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Traits.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

SeedAnalysis::SeedAnalysis(Operation* rootOp) : rootOp{rootOp} {}

std::map<Operation*, unsigned int> SeedAnalysis::getOpDepths() const {

  std::map<Operation*, unsigned int> opDepths;

  rootOp->walk([&](Operation* op) {
    if (!opDepths.count(op)) {
      opDepths[op] = 0;
    }
    for (auto* use : op->getUsers()) {
      if (!opDepths.count(use) || opDepths[op] + 1 > opDepths[use]) {
        opDepths[use] = opDepths[op] + 1;
      }
    }
  });

  return opDepths;
}

std::vector<seed_t> SeedAnalysis::getSeeds(size_t const& width,
                                           std::map<Operation*, unsigned int> const& depthsOf,
                                           SearchMode const& mode) const {

  llvm::StringMap<std::vector<seed_t>> seedsByOpName;

  rootOp->walk([&](Operation* op) {
    auto depth = depthsOf.at(op);
    if (!op->hasTrait<OpTrait::OneResult>() || depth < log2(width)) {
      return;
    }
    bool needsNewSeed = true;
    for (auto& seed : seedsByOpName[op->getName().getStringRef()]) {
      if (seed.size() < width && depthsOf.at(seed.front()) == depth) {
        seed.emplace_back(op);
        needsNewSeed = false;
        break;
      }
    }
    if (needsNewSeed) {
      seed_t seed{op};
      seedsByOpName[op->getName().getStringRef()].emplace_back(seed);
    }
  });

  std::vector<seed_t> seeds;
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
  std::sort(seeds.begin(), seeds.end(), [&](seed_t const& seed1, seed_t const& seed2) {
    if (mode == DefBeforeUse) {
      return depthsOf.at(seed1.front()) < depthsOf.at(seed2.front());
    } else if (mode == UseBeforeDef) {
      return depthsOf.at(seed1.front()) > depthsOf.at(seed2.front());
    } else {
      // Unused so far.
      assert(false);
    }
  });
  return seeds;

}
