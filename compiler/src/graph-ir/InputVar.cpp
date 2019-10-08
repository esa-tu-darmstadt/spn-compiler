//
// Created by ls on 10/7/19.
//

#include "GraphIRNode.h"

InputVar::InputVar(std::string id, int index) : GraphIRNode{std::move(id)}, _index{index} {}

int InputVar::index() const {return _index;}

std::string InputVar::dump() const { return id()+"["+std::to_string(_index)+"]";}