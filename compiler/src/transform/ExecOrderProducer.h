#include <graph-ir/GraphIRNode.h>
#include <stack>
#include "BaseVisitor.h"

enum class NodeKind { Input, Histogram, Product, WeightedSum, Sum, Store };
struct NodeWrapper {
  GraphIRNode* node;
  NodeKind kind;
};

class ExecOrderProducer : public BaseVisitor {

public:
    void produceOrder(const NodeReference& rootNode);

    void visitHistogram(Histogram& n, arg_t arg) override ;

    void visitProduct(Product& n, arg_t arg) override ;

    void visitSum(Sum& n, arg_t arg) override ;

    void visitWeightedSum(WeightedSum& n, arg_t arg) override ;

    void visitInputvar(InputVar& n, arg_t arg) override;

    std::stack<NodeWrapper>& ordered_nodes();

private:
    std::stack<NodeWrapper> _ordered_nodes;
};
