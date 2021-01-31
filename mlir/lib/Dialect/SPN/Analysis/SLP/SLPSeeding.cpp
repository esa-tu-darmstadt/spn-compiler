//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPSeeding.h"
#include <iostream>
#include <llvm/ADT/StringMap.h>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SeedAnalysis::SeedAnalysis(Operation* module) : module{module} {}

std::vector<seed_t> SeedAnalysis::getSeeds(size_t const& width, SPNNodeLevel const& nodeLevels) const {

  std::cout << "Starting seed computation!" << std::endl;

  llvm::StringMap<std::vector<seed_t>> seedsByOpName;

  module->walk([&](Operation* op) {
    op->getLoc().dump();
    auto depth = nodeLevels.getOperationDepth(op);
    if (!op->hasTrait<OpTrait::spn::Vectorizable>() || depth < log2(width)) {
      return;
    }
    bool needsNewSeed = true;
    for (auto& seed : seedsByOpName[op->getName().getStringRef()]) {
      if (seed.size() < width) {
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
  return seeds;

}
