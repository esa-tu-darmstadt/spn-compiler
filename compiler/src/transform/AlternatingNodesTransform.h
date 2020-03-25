
#include <graph-ir/GraphIRNode.h>
#include <unordered_map>
#include "BaseVisitor.h"

class AlternatingNodesTransform : public BaseVisitor {

public:
    void visitHistogram(Histogram& n, arg_t arg) override ;
    void visitGauss(Gauss& n, arg_t arg) override ;

    void visitProduct(Product& n, arg_t arg) override ;

    void visitSum(Sum& n, arg_t arg) override ;

    void visitWeightedSum(WeightedSum& n, arg_t arg) override ;

};

