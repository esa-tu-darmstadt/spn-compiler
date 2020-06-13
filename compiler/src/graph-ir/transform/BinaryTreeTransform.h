//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_BINARYTREETRANSFORM_H
#define SPNC_BINARYTREETRANSFORM_H

#include <graph-ir/GraphIRNode.h>
#include <unordered_map>
#include "BaseVisitor.h"
#include "IRTransformationPass.h"

namespace spnc {

  ///
  /// IRTransformationPass decomposing all operations with more than two inputs
  /// into a tree of two-input operations.
  class BinaryTreeTransform : public BaseVisitor, public IRTransformationPass {

  public:

    using IRTransformationPass::IRTransformationPass;

    void transform(IRGraph& input) override;

    void visitHistogram(Histogram& n, arg_t arg) override;

    void visitGauss(Gauss& n, arg_t arg) override;

    void visitProduct(Product& n, arg_t arg) override;

    void visitSum(Sum& n, arg_t arg) override;

    void visitWeightedSum(WeightedSum& n, arg_t arg) override;

  private:

    NodeReference binarizeTree(const NodeReference rootNode);

    std::unordered_map<std::string, NodeReference> updated_nodes;

    template<class T>
    NodeReference splitChildren(const std::vector<NodeReference>& children, const std::string& prefix);

    NodeReference splitWeightedChildren(const std::vector<WeightedAddend>& children, const std::string& prefix);

  };
}

#endif //SPNC_BINARYTREETRANSFORM_H
