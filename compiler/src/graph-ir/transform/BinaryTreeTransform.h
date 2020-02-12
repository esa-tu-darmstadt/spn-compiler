//
// Created by ls on 10/9/19.
//

#ifndef SPNC_BINARYTREETRANSFORM_H
#define SPNC_BINARYTREETRANSFORM_H


#include <graph-ir/GraphIRNode.h>
#include <unordered_map>
#include "BaseVisitor.h"
#include "IRTransformationPass.h"

namespace spnc {
    class BinaryTreeTransform : public BaseVisitor, public IRTransformationPass {

    public:

        explicit BinaryTreeTransform(ActionWithOutput<IRGraph>& _input);

        IRGraph transform(IRGraph& input) override ;

        NodeReference binarizeTree(const NodeReference& rootNode);

        void visitHistogram(Histogram& n, arg_t arg) override ;

        void visitProduct(Product& n, arg_t arg) override ;

        void visitSum(Sum& n, arg_t arg) override ;

        void visitWeightedSum(WeightedSum& n, arg_t arg) override ;

    private:
        std::unordered_map<std::string, NodeReference> updated_nodes;

        template<class T>
        NodeReference splitChildren(const std::vector<NodeReference>& children, const std::string& prefix) const;

        NodeReference splitWeightedChildren(const std::vector<WeightedAddend>& children, const std::string& prefix) const;

    };
}




#endif //SPNC_BINARYTREETRANSFORM_H
