//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <graph-ir/transform/Visitor.h>
#include <graph-ir/GraphIRNode.h>

using namespace spnc;

Sum::Sum(std::string id, const std::vector<NodeReference>& addends) : GraphIRNode{std::move(id)} {
  std::copy(addends.begin(), addends.end(), std::back_inserter(_addends));
}

const std::vector<NodeReference>& Sum::addends() const { return _addends; }

void Sum::accept(Visitor& visitor, arg_t arg) {
  return visitor.visitSum(*this, arg);
}
