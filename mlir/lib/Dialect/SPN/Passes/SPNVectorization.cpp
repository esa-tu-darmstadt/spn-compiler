//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "SPN/SPNPasses.h"
#include "SPNPassDetails.h"
#include <iostream>
#include <SPN/SPNInterfaces.h>
#include "SPN/Analysis/SLP/SLPTree.h"
using namespace mlir;
using namespace mlir::spn;

namespace {

  struct SPNVectorization : public SPNVectorizationBase<SPNVectorization> {

  protected:
    void runOnOperation() override {
      std::cout << "Starting SPN vectorization..." << std::endl;
      auto func = getOperation();

      func.walk([](Operation* op) {
        if (auto query = dyn_cast<QueryInterface>(op)) {
          for (auto r : query.getRootNodes()) {
            slp::SLPTree graph(r, 4, 3);
          }
        }
      });
    }
  };

}

std::unique_ptr<OperationPass<ModuleOp>> mlir::spn::createSPNVectorizationPass() {
  return std::make_unique<SPNVectorization>();
}

