//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_VISITOR_H
#define SPNC_VISITOR_H

#include <graph-ir/GraphIRNode.h>

namespace spnc {

  using arg_t = std::shared_ptr<void>;

  ///
  /// Visitor for the graph-based IR.
  class Visitor {

  public:

    virtual void visitIRNode(GraphIRNode& n, arg_t arg) = 0;

    virtual void visitInputvar(InputVar& n, arg_t arg) = 0;

    virtual void visitHistogram(Histogram& n, arg_t arg) = 0;

    virtual void visitProduct(Product& n, arg_t arg) = 0;

    virtual void visitSum(Sum& n, arg_t arg) = 0;

    virtual void visitWeightedSum(WeightedSum& n, arg_t arg) = 0;

  };
}

#endif //SPNC_VISITOR_H
