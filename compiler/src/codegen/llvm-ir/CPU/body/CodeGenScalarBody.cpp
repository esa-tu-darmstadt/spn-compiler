//
// Created by lukas on 20.11.19.
//
#include "CodeGenScalarBody.h"
#include <iostream>

namespace spnc {

    Value* CodeGenScalarBody::emitBody(IRGraph& graph, Value* indVar, InputVarValueMap inputs, OutputAddressMap output) {
      // Initialize node-to-value map with inputs.
      for(const auto& inputVar : *graph.inputs){
        node2value[inputVar.get()] = inputs(inputVar->index(), indVar);
      }
      graph.rootNode->accept(*this, nullptr);
      assert(node2value.count(graph.rootNode.get()) && "Node LLVM IR value generated for root node!");
      auto storeAddress = output(indVar);
      builder.CreateStore(node2value[graph.rootNode.get()], storeAddress);
      auto const1 = ConstantInt::get(indVar->getType(), 1);
      return builder.CreateAdd(indVar, const1, "indvar.incr");
    }

    void CodeGenScalarBody::visitHistogram(Histogram &n, arg_t arg) {
      std::vector<Constant*> values;
      size_t max_index = 0;
      for(auto& b : *n.buckets()){
        for(int i=0; i < (b.upperBound - b.lowerBound); ++i){
          values.push_back(ConstantFP::get(getValueType(), b.value));
        }
        max_index = (b.upperBound > max_index) ? b.upperBound : max_index;
      }
      auto arrayType = ArrayType::get(getValueType(), max_index);
      auto initializer = ConstantArray::get(arrayType, values);
      auto globalArray = new GlobalVariable(module, arrayType, true, GlobalValue::InternalLinkage,
                                            initializer, "histogram");
      auto index = getValueForNode(n.indexVar(), arg);
      auto const0 = ConstantInt::get(IntegerType::get(module.getContext(), 32), 0);
      auto address = builder.CreateGEP(globalArray, {const0, index}, "hist_address");
      auto value = builder.CreateLoad(address, "hist_value");
      addMetaData(value, MetadataTag::Histogram);
      node2value[&n] = value;
    }

    void CodeGenScalarBody::visitProduct(Product &n, arg_t arg) {
      assert(n.multiplicands()->size()==2 && "Excepting only binary operations in code generation!");
      auto leftOp = getValueForNode(n.multiplicands()->at(0), arg);
      auto rightOp = getValueForNode(n.multiplicands()->at(1), arg);
      auto product = builder.CreateFMul(leftOp, rightOp, "product");
      addMetaData(product, MetadataTag::Product);
      node2value[&n] = product;
    }

    void CodeGenScalarBody::visitSum(Sum &n, arg_t arg) {
      assert(n.addends()->size()==2 && "Excepting only binary operations in code generation!");
      auto leftOp = getValueForNode(n.addends()->at(0), arg);
      auto rightOp = getValueForNode(n.addends()->at(1), arg);
      auto sum = builder.CreateFAdd(leftOp, rightOp, "sum");
      addMetaData(sum, MetadataTag::Sum);
      node2value[&n] = sum;
    }

    void CodeGenScalarBody::visitWeightedSum(WeightedSum &n, arg_t arg) {
      assert(n.addends()->size()==2 && "Expecting only binary operations in code generation!");
      auto leftAddend = n.addends()->at(0);
      auto leftOp = getValueForNode(leftAddend.addend, arg);
      auto leftConst = ConstantFP::get(getValueType(), leftAddend.weight);
      auto leftMul = builder.CreateFMul(leftOp, leftConst, "left_mul");
      auto rightAddend = n.addends()->at(1);
      auto rightOp = getValueForNode(rightAddend.addend, arg);
      auto rightConst = ConstantFP::get(getValueType(), rightAddend.weight);
      auto rightMul = builder.CreateFMul(rightOp, rightConst, "right_mul");
      auto sum = builder.CreateFAdd(leftMul, rightMul, "weighted_sum");
      addMetaData(leftMul, MetadataTag::WeightedSum);
      addMetaData(rightMul, MetadataTag::WeightedSum);
      addMetaData(sum, MetadataTag::WeightedSum);
      node2value[&n] = sum;
    }

    Type* CodeGenScalarBody::getValueType() {
      return Type::getDoubleTy(module.getContext());
    }

    Value* CodeGenScalarBody::getValueForNode(const NodeReference& node, arg_t arg) {
      if(!node2value.count(node.get())){
        node->accept(*this, std::move(arg));
      }
      return node2value[node.get()];
    }

    void CodeGenScalarBody::addMetaData(Value* val, MetadataTag tag) {
      if (auto *I = dyn_cast<Instruction>(val)) {
        auto metadata = ConstantAsMetadata::get(builder.getInt32(static_cast<int>(tag)));
        auto metadataNode = MDNode::get(builder.getContext(), metadata);
        I->setMetadata("spn.trace.nodeType", metadataNode);
      }
    }
}



