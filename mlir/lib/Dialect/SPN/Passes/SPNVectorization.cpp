//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/SPNPasses.h"
#include "SPNPassDetails.h"
#include <iostream>
#include <mlir/Pass/PassManager.h>
#include "../Analysis/SLP/VectorizationPatterns.h"
#include "SPN/Analysis/SLP/SLPTree.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::spn;

namespace {

  struct SPNVectorization : public SPNVectorizationBase<SPNVectorization> {

  protected:
    void runOnOperation() override {

      std::cout << "Starting SPN vectorization..." << std::endl;
      auto query = getOperation();

      // ============ TREE CHECK ============ //
      query.walk([&](Operation* topLevelOp) {
        for (auto root : query.getRootNodes()) {

          llvm::StringMap<std::vector<Operation*>> operationsByOpCode;
          for (auto& op : root->getBlock()->getOperations()) {
            operationsByOpCode[op.getName().getStringRef().str()].emplace_back(&op);
            auto const& uses = std::distance(op.getUses().begin(), op.getUses().end());
            if (uses > 1) {
              std::cerr << "SPN is not a tree!" << std::endl;
              // TODO: how to handle such cases? and is special handling required at all?
              assert(false);
            }
          }
        }
      });
      // ==================================== //

      auto& depthAnalysis = getAnalysis<SPNNodeLevel>();
      auto& seedAnalysis = getAnalysis<slp::SeedAnalysis>();
      std::cout << "Starting seed computation!" << std::endl;
      auto seeds = seedAnalysis.getSeeds(4, depthAnalysis);
      std::cout << "Seeds computed" << std::endl;

      if (!seeds.empty()) {
        slp::SLPTree graph(seeds.front(), 3);
        OwningRewritePatternList patterns;
        std::vector<std::vector<Operation*>> vectors;
        for (auto const& node_ptr : graph.getNodes()) {
          for (size_t i = 0; i < node_ptr->numVectors(); ++i) {
            vectors.emplace_back(node_ptr->getVector(i));
          }
        }
        slp::populateVectorizationPatterns(patterns, &getContext(), vectors);
        auto op = getOperation();
        FrozenRewritePatternList frozenPatterns(std::move(patterns));
        applyPatternsAndFoldGreedily(op, frozenPatterns);
      }

    }

  };

}

std::unique_ptr<OperationPass<JointQuery>> mlir::spn::createSPNVectorizationPass() {
  return std::make_unique<SPNVectorization>();
}

