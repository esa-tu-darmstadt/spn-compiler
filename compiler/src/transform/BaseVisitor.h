//
// Created by ls on 10/9/19.
//

#ifndef SPNC_BASEVISITOR_H
#define SPNC_BASEVISITOR_H

#include "Visitor.h"

class BaseVisitor : public Visitor {

public:

    void visitIRNode(GraphIRNode& n, arg_t arg) override ;

    void visitInputvar(InputVar& n, arg_t arg) override ;

    void visitHistogram(Histogram& n, arg_t arg) override ;
  
    void visitGauss(Gauss& n, arg_t arg) override ;

    void visitProduct(Product& n, arg_t arg) override ;

    void visitSum(Sum& n, arg_t arg) override ;

    void visitWeightedSum(WeightedSum& n, arg_t arg) override ;

};


#endif //SPNC_BASEVISITOR_H
