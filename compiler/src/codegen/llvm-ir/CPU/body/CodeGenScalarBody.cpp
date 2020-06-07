//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "CodeGenScalarBody.h"

using namespace spnc;

Value* CodeGenScalarBody::emitBody(IRGraph& graph, Value* indVar, InputVarValueMap inputs, OutputAddressMap output, const Configuration& config) {
  // Initialize node-to-value map with inputs.
  for (auto inputVar : graph.inputs()) {
    node2value[inputVar->id()] = inputs(inputVar->index(), indVar);
  }
  // Recursively visit the SPN graph starting at the root node.
  graph.rootNode()->accept(*this, nullptr);
  assert(node2value.count(graph.rootNode()->id()) && "No LLVM IR value generated for root node!");
  // Store the result into the output store location.
  auto storeAddress = output(indVar);
  builder.CreateStore(node2value[graph.rootNode()->id()], storeAddress);
  // Increment the loop induction variable.
  auto const1 = ConstantInt::get(indVar->getType(), 1);
  return builder.CreateAdd(indVar, const1, "indvar.incr");
}

void CodeGenScalarBody::visitHistogram(Histogram& n, arg_t arg) {
  std::vector<Constant*> values;
  size_t max_index = 0;
  // Flatten the histogram into a array of doubles that can simply
  // be indexed with the feature value.
  for (auto& b : n.buckets()) {
    for (int i = 0; i < (b.upperBound - b.lowerBound); ++i) {
      values.push_back(ConstantFP::get(getValueType(), b.value));
    }
    max_index = (b.upperBound > max_index) ? b.upperBound : max_index;
  }
  // Create the histogram array as global constant array.
  auto arrayType = ArrayType::get(getValueType(), max_index);
  auto initializer = ConstantArray::get(arrayType, values);
  auto globalArray = new GlobalVariable(module, arrayType, true, GlobalValue::InternalLinkage,
                                        initializer, "histogram");
  // Index and load the value using the feature input value.
  auto index = getValueForNode(&n.indexVar(), arg);
  auto const0 = ConstantInt::get(IntegerType::get(module.getContext(), 32), 0);
  auto address = builder.CreateGEP(globalArray, {const0, index}, "hist_address");
  auto value = builder.CreateLoad(address, "hist_value");
  addMetaData(value, TraceMDTag::Histogram);
  node2value[n.id()] = value;
}

void CodeGenScalarBody::visitProduct(Product& n, arg_t arg) {
  assert(n.multiplicands().size() == 2 && "Excepting only binary operations in code generation!");
  auto leftOp = getValueForNode(n.multiplicands()[0], arg);
  auto rightOp = getValueForNode(n.multiplicands()[1], arg);
  auto product = builder.CreateFMul(leftOp, rightOp, "product");
  addMetaData(product, TraceMDTag::Product);
  node2value[n.id()] = product;
}

void CodeGenScalarBody::visitSum(Sum& n, arg_t arg) {
  assert(n.addends().size() == 2 && "Excepting only binary operations in code generation!");
  auto leftOp = getValueForNode(n.addends()[0], arg);
  auto rightOp = getValueForNode(n.addends()[1], arg);
  auto sum = builder.CreateFAdd(leftOp, rightOp, "sum");
  addMetaData(sum, TraceMDTag::Sum);
  node2value[n.id()] = sum;
}

void CodeGenScalarBody::visitWeightedSum(WeightedSum& n, arg_t arg) {
  assert(n.addends().size() == 2 && "Expecting only binary operations in code generation!");
  // Multiply both operands with their associated weight.
  auto leftAddend = n.addends()[0];
  auto leftOp = getValueForNode(leftAddend.addend, arg);
  auto leftConst = ConstantFP::get(getValueType(), leftAddend.weight);
  auto leftMul = builder.CreateFMul(leftOp, leftConst, "left_mul");
  auto rightAddend = n.addends()[1];
  auto rightOp = getValueForNode(rightAddend.addend, arg);
  auto rightConst = ConstantFP::get(getValueType(), rightAddend.weight);
  auto rightMul = builder.CreateFMul(rightOp, rightConst, "right_mul");
  // Sum up the two multiplication results.
  auto sum = builder.CreateFAdd(leftMul, rightMul, "weighted_sum");
  addMetaData(leftMul, TraceMDTag::WeightedSum);
  addMetaData(rightMul, TraceMDTag::WeightedSum);
  addMetaData(sum, TraceMDTag::WeightedSum);
  node2value[n.id()] = sum;
}

Type* CodeGenScalarBody::getValueType() {
  return Type::getDoubleTy(module.getContext());
}

Value* CodeGenScalarBody::getValueForNode(NodeReference node, arg_t arg) {
  if (!node2value.count(node->id())) {
    node->accept(*this, std::move(arg));
  }
  return node2value[node->id()];
}

void CodeGenScalarBody::addMetaData(Value* val, TraceMDTag tag) {
  // Attach meta-data to each instruction indicating from which type of SPN node it was created
  // for identification in later stages.
  if (auto* I = dyn_cast<Instruction>(val)) {
    auto metadata = ConstantAsMetadata::get(builder.getInt32(static_cast<ushort>(tag)));
    auto metadataNode = MDNode::get(builder.getContext(), metadata);
    I->setMetadata(TraceMDName, metadataNode);
  }
}



