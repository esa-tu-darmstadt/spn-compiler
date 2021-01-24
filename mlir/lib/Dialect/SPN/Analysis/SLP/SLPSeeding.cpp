//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPSeeding.h"
#include <iostream>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SeedAnalysis::SeedAnalysis(Operation* module) : operationsByName{} {

  module->walk([&](Operation* op) {
    if (op->hasTrait<OpTrait::spn::Vectorizable>() || op->hasTrait<OpTrait::spn::Binarizable>()) {
      operationsByName[op->getName().getStringRef()].emplace_back(op);
    }
  });

  // Sort operations by their number of operands in descending order to maximize vectorization tree sizes.
  for (auto& entry : operationsByName) {
    std::sort(std::begin(entry.second), std::end(entry.second), [&](Operation* a, Operation* b) {
      return a->getNumOperands() > b->getNumOperands();
    });
  }
}

std::vector<seed_t> SeedAnalysis::getSeeds(size_t const& width) const {

  std::cout << "Starting seed computation!" << std::endl;

  std::vector<seed_t> seeds;

  for (auto& entry : operationsByName) {

    auto& operations = entry.second;
    SearchMode searchMode = GREEDY;
    std::vector<bool> assignedOps(operations.size());
    for (size_t i = 0; i < operations.size() - width; ++i) {

      if (assignedOps.at(i)) {
        continue;
      }

      // Begin a new seed.
      seed_t seed;
      auto& firstOp = operations.at(i);
      seed.emplace_back(firstOp);
      assignedOps.at(i) = true;

      for (size_t j = i + 1; j < operations.size() && seed.size() < width; ++j) {

        if (assignedOps.at(j)) {
          continue;
        }

        auto& nextOp = operations.at(j);
        // When in SIZE mode, simply add the next biggest operation to the seed.
        if (searchMode == GREEDY) {
          seed.emplace_back(nextOp);
          assignedOps.at(j) = true;
        }

        if (seed.size() == width) {
          break;
        }
      }
      if (seed.size() == width) {
        seeds.emplace_back(seed);
      } else {
        searchMode = FAILED;
      }
    }

  }

  return seeds;

}
