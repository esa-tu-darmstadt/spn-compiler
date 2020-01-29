#include "GraphIRTools.h"
#include "codegen/shared/BFSOrderProducer.h"


std::vector<NodeReference> getAtLeastNEqualHeightNodes(GraphIRNode* root, size_t n) {
  // Perform BFS to find starting tree level
  std::vector<NodeReference> vectorRoots;
  BFSOrderProducer visitor;
  root->accept(visitor, {});
  while (!visitor.q.empty()) {
    if (visitor.currentLevel < visitor.q.front().first) {
      if (vectorRoots.size() >= n) {
        break;
      } else {
        visitor.currentLevel++;
	vectorRoots.clear();
      }
    }
    visitor.q.front().second->accept(visitor, {});
    vectorRoots.push_back(visitor.q.front().second);
    visitor.q.pop();
  }

  if (vectorRoots.size() < n)
    return {};
  return vectorRoots;
}

size_t getInputLength(WeightedSum &n) { return n.addends()->size(); }

size_t getInputLength(Sum &n) { return n.addends()->size(); }

size_t getInputLength(Product &n) { return n.multiplicands()->size(); }

NodeReference getInput(WeightedSum &n, int pos) {
  return (*n.addends())[pos].addend;
}
NodeReference getInput(Sum &n, int pos) { return (*n.addends())[pos]; }
NodeReference getInput(Product &n, int pos) {
  return (*n.multiplicands())[pos];
}
