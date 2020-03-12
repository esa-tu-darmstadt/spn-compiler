//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <graph-ir/transform/Visitor.h>
#include <graph-ir/GraphIRNode.h>

using namespace spnc;

WeightedSum::WeightedSum(std::string id, const std::vector<WeightedAddend>& addends) : GraphIRNode(std::move(id)) {
  std::copy(addends.begin(), addends.end(), std::back_inserter(_addends));
}

const std::vector<WeightedAddend>& WeightedSum::addends() const { return _addends; }

void WeightedSum::accept(Visitor& visitor, arg_t arg) {
  return visitor.visitWeightedSum(*this, arg);
}

