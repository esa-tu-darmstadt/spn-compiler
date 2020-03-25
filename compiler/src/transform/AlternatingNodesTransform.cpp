
#include "AlternatingNodesTransform.h"
#include <queue>
template<class T>
class sameKindChecker : public BaseVisitor {

public:
  void visitHistogram(Histogram &n, arg_t arg) { isSame = false; }
  void visitGauss(Gauss &n, arg_t arg) { isSame = false; }

  void visitProduct(Product &n, arg_t arg) {
    setFlag(n);
  }

  void visitSum(Sum &n, arg_t arg) {
    setFlag(n);
  }

  void visitWeightedSum(WeightedSum &n, arg_t arg) {
    setFlag(n);
  }

  void setFlag(T n) {
    isSame = true;
  }

  template<class U>
  void setFlag(U n) {
    isSame = false;
  }
  bool isSame;
};

void AlternatingNodesTransform::visitHistogram(Histogram &n, arg_t arg) {
}

void AlternatingNodesTransform::visitGauss(Gauss &n, arg_t arg) {
}

void AlternatingNodesTransform::visitProduct(Product &n, arg_t arg) {
  sameKindChecker<Product> checker;
  auto newMultiplicands = std::make_shared<std::vector<NodeReference>>();
  std::queue<NodeReference> childrenToCheck;
  for (auto& child : *n.multiplicands()) {
    childrenToCheck.push(child);
  }

  while (!childrenToCheck.empty()) {
    auto& child = childrenToCheck.front();
    child->accept(checker, {});
    if (checker.isSame == false) {
      newMultiplicands->push_back(child);
    } else {
      Product* sameKindChild = (Product*) child.get();
      for (auto& childChild : *sameKindChild->multiplicands()) {
	childrenToCheck.push(childChild);
      }
    }
    childrenToCheck.pop();
  }
  n.setMultiplicands(newMultiplicands);
  
  for (auto& child : *n.multiplicands()) {
    child->accept(*this, {});
  }
}

void AlternatingNodesTransform::visitSum(Sum &n, arg_t arg) {
  sameKindChecker<Sum> checker;
  auto newAddends = std::make_shared<std::vector<NodeReference>>();
  std::queue<NodeReference> childrenToCheck;
  for (auto& child : *n.addends()) {
    childrenToCheck.push(child);
  }

  while (!childrenToCheck.empty()) {
    auto& child = childrenToCheck.front();
    child->accept(checker, {});
    if (checker.isSame == false) {
      newAddends->push_back(child);
    } else {
      Sum* sameKindChild = (Sum*) child.get();
      for (auto& childChild : *sameKindChild->addends()) {
	childrenToCheck.push(childChild);
      }
    }
    childrenToCheck.pop();
  }
  n.setAddends(newAddends);
  
  for (auto& child : *n.addends()) {
    child->accept(*this, {});
  }
}

void AlternatingNodesTransform::visitWeightedSum(WeightedSum& n, arg_t arg){
  sameKindChecker<WeightedSum> checker;
  auto newAddends = std::make_shared<std::vector<WeightedAddend>>();
  std::queue<WeightedAddend> childrenToCheck;
  for (auto& child : *n.addends()) {
    childrenToCheck.push(child);
  }

  while (!childrenToCheck.empty()) {
    auto& child = childrenToCheck.front();
    child.addend->accept(checker, {});
    if (checker.isSame == false) {
      newAddends->push_back(child);
    } else {
      WeightedSum* sameKindChild = (WeightedSum*) child.addend.get();
      for (auto& childChild : *sameKindChild->addends()) {
	childrenToCheck.push({childChild.addend, childChild.weight*child.weight});
      }
    }
    childrenToCheck.pop();
  }
  n.setAddends(newAddends);
  
  for (auto& child : *n.addends()) {
    child.addend->accept(*this, {});
  }
}
