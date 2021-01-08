//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPGraph.h"
#include "SPN/SPNInterfaces.h"

#include <iostream>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SLPGraph::SLPGraph(Operation* root) : seeds{} {
  assert(root);
  analyzeGraph(root);
}

const SmallPtrSet<Operation, 16>& SLPGraph::getSeeds() const { return seeds; }

void SLPGraph::analyzeGraph(Operation* root) {
  if (auto query = dyn_cast<QueryInterface>(root)) {
    // If this is a query op, traverse all root nodes stored in the query.
    for (auto r : query.getRootNodes()) {
      analyzeGraph(r);
    }

  } else {
    traverseSubgraph(root);
  }
}

void SLPGraph::traverseSubgraph(Operation* root) {
  std::cout << root->getName().getStringRef().str() << std::endl;
  if (auto leaf = dyn_cast<LeafNodeInterface>(root)) {
    std::cout << "\tis a leaf." << std::endl;
  } else {
    std::cout << "\tis an inner node." << std::endl;
    for (auto op : root->getOperands()) {
      traverseSubgraph(op.getDefiningOp());
    }
  }

}
