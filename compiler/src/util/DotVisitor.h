//
// Created by ls on 10/9/19.
//

#ifndef SPNC_DOTVISITOR_H
#define SPNC_DOTVISITOR_H


#include <transform/BaseVisitor.h>
#include <sstream>

class DotVisitor : BaseVisitor {

public:

    void writeDotGraph(const NodeReference& rootNode, const std::string& outputFile);

    void visitInputvar(InputVar& n, arg_t arg) override ;

    void visitHistogram(Histogram& n, arg_t arg) override ;
    void visitGauss(Gauss& n, arg_t arg) override ;

    void visitProduct(Product& n, arg_t arg) override ;

    void visitSum(Sum& n, arg_t arg) override ;

    void visitWeightedSum(WeightedSum& n, arg_t arg) override ;

private:
    std::stringstream nodes{};
    std::stringstream edges{};
};


#endif //SPNC_DOTVISITOR_H
