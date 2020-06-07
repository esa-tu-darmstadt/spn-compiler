#include <graph-ir/GraphIRNode.h>

using namespace spnc;

std::vector<NodeReference> getAtLeastNEqualHeightNodes(GraphIRNode* root, size_t n);

size_t getInputLength(WeightedSum &n);

size_t getInputLength(Sum &n);

size_t getInputLength(Product &n);

NodeReference getInput(WeightedSum &n, int pos);
NodeReference getInput(Sum &n, int pos);
NodeReference getInput(Product &n, int pos);
