//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <graph-ir/GraphIRNode.h>

using namespace spnc;

GraphIRNode::GraphIRNode(std::string id) : _id(std::move(id)) {}

std::string GraphIRNode::dump() const { return _id; }

std::string GraphIRNode::id() const { return _id; }

std::ostream& operator<<(std::ostream& os, const GraphIRNode& node) {
  os << node.dump();
  return os;
}
