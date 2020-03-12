//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <graph-ir/transform/Visitor.h>
#include <graph-ir/GraphIRNode.h>

using namespace spnc;

InputVar::InputVar(std::string id, int index) : GraphIRNode{std::move(id)}, _index{index} {}

int InputVar::index() const { return _index; }

std::string InputVar::dump() const { return id() + "[" + std::to_string(_index) + "]"; }

void InputVar::accept(Visitor& visitor, arg_t arg) {
  return visitor.visitInputvar(*this, arg);
}
