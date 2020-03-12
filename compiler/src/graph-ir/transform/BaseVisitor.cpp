//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <iostream>
#include "BaseVisitor.h"
#include <util/Logging.h>

using namespace spnc;

void BaseVisitor::visitIRNode(GraphIRNode& n, arg_t arg) {
  SPNC_FATAL_ERROR("Fall-through to non-implemented base case");
}

void BaseVisitor::visitInputvar(InputVar& n, arg_t arg) {
  return visitIRNode(n, arg);
}

void BaseVisitor::visitHistogram(Histogram& n, arg_t arg) {
  return visitIRNode(n, arg);
}

void BaseVisitor::visitProduct(Product& n, arg_t arg) {
  return visitIRNode(n, arg);
}

void BaseVisitor::visitSum(Sum& n, arg_t arg) {
  return visitIRNode(n, arg);
}

void BaseVisitor::visitWeightedSum(WeightedSum& n, arg_t arg) {
  return visitIRNode(n, arg);
}

