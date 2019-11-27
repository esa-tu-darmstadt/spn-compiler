#include "ExecOrderProducer.h"

void ExecOrderProducer::produceOrder(const NodeReference &rootNode) {
  _ordered_nodes.push({&*rootNode, NodeKind::Store});
  rootNode->accept(*this, {});
}

void ExecOrderProducer::visitInputvar(InputVar &n, arg_t arg) {
  _ordered_nodes.push({&n, NodeKind::Input});
}

void ExecOrderProducer::visitHistogram(Histogram &n, arg_t arg) {
  _ordered_nodes.push({&n, NodeKind::Histogram});
  n.indexVar()->accept(*this, {});
}

void ExecOrderProducer::visitProduct(Product &n, arg_t arg) {
  _ordered_nodes.push({&n, NodeKind::Product});
  for (auto &c : *n.multiplicands()) {
    c->accept(*this, {});
  }
}

void ExecOrderProducer::visitSum(Sum &n, arg_t arg) {
  _ordered_nodes.push({&n, NodeKind::Sum});
  for (auto &c : *n.addends()) {
    c->accept(*this, {});
  }
}

void ExecOrderProducer::visitWeightedSum(WeightedSum &n, arg_t arg) {
  _ordered_nodes.push({&n, NodeKind::WeightedSum});
  for (auto &c : *n.addends()) {
    c.addend->accept(*this, {});
  }
}

std::stack<NodeWrapper>& ExecOrderProducer::ordered_nodes() {
  return _ordered_nodes;
}
