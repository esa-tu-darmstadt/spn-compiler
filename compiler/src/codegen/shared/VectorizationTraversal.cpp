#include "VectorizationTraversal.h"

std::vector<std::pair<std::string, std::vector<NodeReference>>> VectorizationTraversal::collectPaths(const NodeReference &rootNode) {
  _paths.clear();
  auto path = std::make_shared<std::pair<std::string, std::vector<NodeReference>>>();
  path->second.push_back(rootNode);
  rootNode->accept(*this, path);
  return _paths;
}

void VectorizationTraversal::visitHistogram(Histogram &n, arg_t arg) {

  if (_pruned.find(n.id()) != _pruned.end())
    return;
  
  std::pair<std::string, std::vector<NodeReference>>* path =
      (std::pair<std::string, std::vector<NodeReference>> *)arg.get();
  
  path->second.push_back(n.indexVar());
  path->first = path->first + "h";
  _paths.push_back(*path);
}

void VectorizationTraversal::visitProduct(Product &n, arg_t arg) {
  
  if (_pruned.find(n.id()) != _pruned.end())
    return;
  
  std::pair<std::string, std::vector<NodeReference>> path =
      *(std::pair<std::string, std::vector<NodeReference>> *)arg.get();

  path.first = path.first + "m";
  
  for (auto &c : *n.multiplicands()) {
    auto p = std::make_shared<std::pair<std::string, std::vector<NodeReference>>>(path);
    p->second.push_back(c);
    c->accept(*this, p);
  }
}

void VectorizationTraversal::visitSum(Sum &n, arg_t arg) {
  
  if (_pruned.find(n.id()) != _pruned.end())
    return;
  
  std::pair<std::string, std::vector<NodeReference>> path =
      *(std::pair<std::string, std::vector<NodeReference>> *)arg.get();

  path.first = path.first + "s";
  
  for (auto &c : *n.addends()) {
    auto p = std::make_shared<std::pair<std::string, std::vector<NodeReference>>>(path);
    p->second.push_back(c);
    c->accept(*this, p);
  }
}

void VectorizationTraversal::visitWeightedSum(WeightedSum &n, arg_t arg) {
  // TODO: Evaluate merging mul of weight into children (which are always (?) muls) vs using FMA - trade off
  
  if (_pruned.find(n.id()) != _pruned.end())
    return;
  
  std::pair<std::string, std::vector<NodeReference>> path =
      *(std::pair<std::string, std::vector<NodeReference>> *)arg.get();

  path.first = path.first + "w";
  
  for (auto &c : *n.addends()) {
    auto p = std::make_shared<std::pair<std::string, std::vector<NodeReference>>>(path);
    p->second.push_back(c.addend);
    c.addend->accept(*this, p);
  }
}
