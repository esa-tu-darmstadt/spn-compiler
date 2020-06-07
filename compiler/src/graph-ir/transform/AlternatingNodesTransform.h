//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_ALTERNATINGNODESTRANSFORM_H
#define SPNC_ALTERNATINGNODESTRANSFORM_H


#include <graph-ir/GraphIRNode.h>
#include <unordered_map>
#include "BaseVisitor.h"
#include "IRTransformationPass.h"

using namespace spnc;
class AlternatingNodesTransform : public BaseVisitor, public IRTransformationPass {

public:
  
    using IRTransformationPass::IRTransformationPass;
  
    void transform(IRGraph& input) override;
  
    void visitHistogram(Histogram& n, arg_t arg) override ;
    void visitGauss(Gauss& n, arg_t arg) override ;

    void visitProduct(Product& n, arg_t arg) override ;

    void visitSum(Sum& n, arg_t arg) override ;

    void visitWeightedSum(WeightedSum& n, arg_t arg) override ;

};

#endif
