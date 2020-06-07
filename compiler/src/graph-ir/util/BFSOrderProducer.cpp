#include "BFSOrderProducer.h"
void BFSOrderProducer::visitInputvar(InputVar &n, arg_t arg) {}
void BFSOrderProducer::visitHistogram(Histogram &n, arg_t arg) {
  q.push({currentLevel + 1, &n.indexVar()});
}
void BFSOrderProducer::visitGauss(Gauss &n, arg_t arg) {
  q.push({currentLevel + 1, &n.indexVar()});
}
void BFSOrderProducer::visitProduct(Product &n, arg_t arg) {
  for (auto &c : n.multiplicands()) {
    q.push({currentLevel + 1, c});
  }
}

void BFSOrderProducer::visitSum(Sum &n, arg_t arg) {
  for (auto &c : n.addends()) {
    q.push({currentLevel + 1, c});
  }
}

void BFSOrderProducer::visitWeightedSum(WeightedSum &n, arg_t arg) {
  for (auto &c : n.addends()) {
    q.push({currentLevel + 1, c.addend});
  }
}
