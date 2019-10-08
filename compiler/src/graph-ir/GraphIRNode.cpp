//
// Created by ls on 10/7/19.
//

#include "GraphIRNode.h"

GraphIRNode::GraphIRNode(std::string id) : _id(std::move(id)){}

std::string GraphIRNode::dump() const {return _id;}

std::string GraphIRNode::id() const {return _id;}

std::ostream& operator<<(std::ostream &os, const GraphIRNode &node) {
    os << node.dump();
    return os;
}
