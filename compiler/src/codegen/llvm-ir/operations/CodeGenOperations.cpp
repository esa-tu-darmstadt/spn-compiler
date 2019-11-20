//
// Created by lukas on 18.11.19.
//

#include "CodeGenOperations.h"

CodeGenOperations::CodeGenOperations(Module& m, Function& f, IRBuilder<> &b,
        InputVarValueMap inputMap) : module{m}, function{f}, builder{b},
        inputVarValueMap{inputMap} {}

Type* CodeGenOperations::getValueType() {
    return Type::getDoubleTy(module.getContext());
}

Value* CodeGenOperations::getValueForNode(NodeReference node, arg_t arg) {
    if(!node2value.count(node.get())){
        node->accept(*this, arg);
    }
    return node2value[node.get()];
}

void CodeGenOperations::visitInputvar(InputVar &n, arg_t arg) {
    node2value[&n] = inputVarValueMap(n.index(), builder, module.getContext());
}

void CodeGenOperations::visitHistogram(Histogram &n, arg_t arg) {
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
    node2value[&n] = value;
}

void CodeGenOperations::visitProduct(Product &n, arg_t arg) {
    assert(n.multiplicands()->size()==2 && "Excepting only binary operations in code generation!");
    auto leftOp = getValueForNode(n.multiplicands()->at(0), arg);
    auto rightOp = getValueForNode(n.multiplicands()->at(1), arg);
    auto product = builder.CreateFMul(leftOp, rightOp, "product");
    node2value[&n] = product;
}

void CodeGenOperations::visitSum(Sum &n, arg_t arg) {
    assert(n.addends()->size()==2 && "Excepting only binary operations in code generation!");
    auto leftOp = getValueForNode(n.addends()->at(0), arg);
    auto rightOp = getValueForNode(n.addends()->at(1), arg);
    auto sum = builder.CreateFAdd(leftOp, rightOp, "sum");
    node2value[&n] = sum;
}

void CodeGenOperations::visitWeightedSum(WeightedSum &n, arg_t arg) {
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
    node2value[&n] = sum;
}