//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPNode.h"

#include <iostream>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SLPNode::SLPNode(size_t const& width) : width{width}, operations{} {
}

SLPNode::SLPNode(std::vector<Operation*> const& values) : width{values.size()}, operations{values} {
}


