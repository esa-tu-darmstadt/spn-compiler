//
// Created by lukas on 21.11.19.
//

#ifndef SPNC_CODEGENSCALARBODY_H
#define SPNC_CODEGENSCALARBODY_H

#include <unordered_map>
#include <transform/BaseVisitor.h>
#include "CodeGenBody.h"

namespace spnc {
    
    class CodeGenScalarBody : public CodeGenBody, BaseVisitor {

    public:
        CodeGenScalarBody(Module& m, Function& f, IRBuilder<>& b) : CodeGenBody(m, f, b) {}

        Value* emitBody(IRGraph& graph, Value* indVar, InputVarValueMap inputs, OutputAddressMap output) override;

        void visitHistogram(Histogram &n, arg_t arg) override;

        void visitProduct(Product &n, arg_t arg) override;

        void visitSum(Sum &n, arg_t arg) override;

        void visitWeightedSum(WeightedSum &n, arg_t arg) override;
    private:
        std::unordered_map<GraphIRNode*, Value*> node2value;

        Type* getValueType();

        Value* getValueForNode(const NodeReference& node, arg_t arg);
    };
}



#endif //SPNC_CODEGENSCALARBODY_H
