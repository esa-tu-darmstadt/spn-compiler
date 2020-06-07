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

    /// Visit generic graph-IR node.
    /// \param n Generic graph-IR node.
    /// \param arg Pass-through argument.
    virtual void visitIRNode(GraphIRNode& n, arg_t arg) = 0;

    /// Visit feature (input variable) node.
    /// \param n SPN feature node.
    /// \param arg Pass-through argument.
    virtual void visitInputvar(InputVar& n, arg_t arg) = 0;

    /// Visit histogram.
    /// \param n Histogram as SPN leaf node.
    /// \param arg Pass-through argument.
    virtual void visitHistogram(Histogram& n, arg_t arg) = 0;
    
    virtual void visitGauss(Gauss &n, arg_t arg) = 0;

    /// Visit product node.
    /// \param n N-ary product.
    /// \param arg Pass-through argument.
    virtual void visitProduct(Product& n, arg_t arg) = 0;

    /// Visit n-ary, non-weighted sum.
    /// \param n N-ary, non-weighted sum.
    /// \param arg Pass-through argument.
    virtual void visitSum(Sum& n, arg_t arg) = 0;

    /// Visit n-ary, weighted sum.
    /// \param n N-ary weighted sum.
    /// \param arg Pass-through argument.
    virtual void visitWeightedSum(WeightedSum& n, arg_t arg) = 0;

  };
}

#endif //SPNC_VISITOR_H
