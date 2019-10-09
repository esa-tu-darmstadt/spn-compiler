//
// Created by ls on 10/9/19.
//

#ifndef SPNC_VISITOR_H
#define SPNC_VISITOR_H

#include "../graph-ir/GraphIRNode.h"

typedef std::shared_ptr<void> arg_t;

class Visitor {

public:

    virtual void visitIRNode(GraphIRNode& n, arg_t arg) = 0;

    virtual void visitInputvar(InputVar& n, arg_t arg) = 0;

    virtual void visitHistogram(Histogram& n, arg_t arg) = 0;

    virtual void visitProduct(Product& n, arg_t arg) = 0;

    virtual void visitSum(Sum& n, arg_t arg) = 0;

    virtual void visitWeightedSum(WeightedSum& n, arg_t arg) = 0;

};

#endif //SPNC_VISITOR_H
