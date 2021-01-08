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
#include "SPN/Analysis/SLP/SLPGraph.h"
using namespace mlir;
using namespace mlir::spn;

namespace {

  struct SPNVectorization : public SPNVectorizationBase<SPNVectorization> {

  protected:
    void runOnOperation() override {
      // TODO
      std::cout << "Starting SPN vectorization..." << std::endl;
      auto module = getOperation();
      llvm::SmallVector<Operation*, 5> queries;
      module.walk([&queries](Operation* op) {
        if (auto query = dyn_cast<QueryInterface>(op)) {
          queries.push_back(op);
        }
      });
      slp::SLPGraph graph(queries.front());
      std::cout << queries.front()->getName().getStringRef().str() << std::endl;

    }
  };

}

std::unique_ptr<OperationPass<ModuleOp>> mlir::spn::createSPNVectorizationPass() {
  return std::make_unique<SPNVectorization>();
}

