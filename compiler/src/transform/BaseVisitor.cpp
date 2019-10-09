//
// Created by ls on 10/9/19.
//

#include <iostream>
#include <cassert>
#include "BaseVisitor.h"

void BaseVisitor::visitIRNode(GraphIRNode& n, arg_t arg) {
    std::cerr << "Fall-through to non-implemented base case" << std::endl;
    assert(false);
}

void BaseVisitor::visitInputvar(InputVar& n, arg_t arg) {
    return visitIRNode(n, arg);
}

void BaseVisitor::visitHistogram(Histogram& n, arg_t arg){
    return visitIRNode(n, arg);
}

void BaseVisitor::visitProduct(Product& n, arg_t arg) {
    return visitIRNode(n, arg);
}

void BaseVisitor::visitSum(Sum& n, arg_t arg){
    return visitIRNode(n, arg);
}

void BaseVisitor::visitWeightedSum(WeightedSum& n, arg_t arg){
    return visitIRNode(n, arg);
}