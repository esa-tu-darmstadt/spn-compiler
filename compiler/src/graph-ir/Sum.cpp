//
// Created by ls on 10/8/19.
//
#include "GraphIRNode.h"
#include "../transform/Visitor.h"

Sum::Sum(std::string id, const std::vector<NodeReference> &addends) : GraphIRNode{std::move(id)}{
    _addends = std::make_shared<std::vector<NodeReference>>(addends.size());
    std::copy(addends.begin(), addends.end(), _addends->begin());
}

std::shared_ptr<std::vector<NodeReference>> Sum::addends() const {return _addends;}

void Sum::accept(Visitor& visitor, arg_t arg) {
    return visitor.visitSum(*this, arg);
}