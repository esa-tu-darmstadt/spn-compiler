//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <iostream>
#include "MLIRGraphGen.h"
#include "SPN/SPNDialect.h"
#include "SPN/SPNOps.h"

using namespace spnc;
using namespace mlir::spn;

MLIRGraphGen::MLIRGraphGen(OpBuilder& _builder, std::unordered_map<std::string, Value>& n2v)
    : builder{_builder}, node2value{n2v} {}

void MLIRGraphGen::visitProduct(spnc::Product& n, spnc::arg_t arg) {
  llvm::SmallVector<Value, 10> ops;
  for (auto& o : n.multiplicands()) {
    ops.push_back(getValueForNode(o, arg));
  }
  auto product = builder.create<ProductOp>(builder.getUnknownLoc(), ops);
  node2value[n.id()] = product;
}

void MLIRGraphGen::visitSum(spnc::Sum& n, spnc::arg_t arg) {
  llvm::SmallVector<Value, 10> ops;
  for (auto& o : n.addends()) {
    ops.push_back(getValueForNode(o, arg));
  }
  auto sum = builder.create<SumOp>(builder.getUnknownLoc(), ops);
  node2value[n.id()] = sum;
}

void MLIRGraphGen::visitWeightedSum(spnc::WeightedSum& n, spnc::arg_t arg) {
  llvm::SmallVector<Value, 10> ops;
  llvm::SmallVector<double, 10> weights;
  for (auto& wa : n.addends()) {
    ops.push_back(getValueForNode(wa.addend, arg));
    weights.push_back(wa.weight);
  }
  auto sum = builder.create<WeightedSumOp>(builder.getUnknownLoc(), ops, weights);
  node2value[n.id()] = sum;
}

void MLIRGraphGen::visitHistogram(spnc::Histogram& n, spnc::arg_t arg) {
  auto indexVar = getValueForNode(&n.indexVar(), arg);
  llvm::SmallVector<bucket_t, 256> buckets;
  for (auto b : n.buckets()) {
    buckets.push_back(std::tie(b.lowerBound, b.upperBound, b.value));
  }
  auto histogram = builder.create<HistogramOp>(builder.getUnknownLoc(), indexVar, buckets);
  node2value[n.id()] = histogram;
}

Value MLIRGraphGen::getValueForNode(NodeReference node, arg_t arg) {
  if (!node2value.count(node->id())) {
    node->accept(*this, std::move(arg));
  }
  return node2value.at(node->id());
}