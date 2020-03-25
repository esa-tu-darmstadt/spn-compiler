//
// Created by ls on 10/8/19.
//
#include "GraphIRNode.h"
#include "../transform/Visitor.h"

Product::Product(std::string id, const std::vector<NodeReference> &multiplicands) : GraphIRNode{std::move(id)}{
    _multiplicands = std::make_shared<std::vector<NodeReference>>(multiplicands.size());
    std::copy(multiplicands.begin(), multiplicands.end(), _multiplicands->begin());
}

std::shared_ptr<std::vector<NodeReference>> Product::multiplicands() { return _multiplicands; }

void Product::setMultiplicands(std::shared_ptr<std::vector<NodeReference>> newMultiplicands) {
  _multiplicands = newMultiplicands;
}

void Product::accept(Visitor& visitor, arg_t arg) {
    return visitor.visitProduct(*this, arg);
}
