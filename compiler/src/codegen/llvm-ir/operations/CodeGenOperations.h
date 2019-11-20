//
// Created by lukas on 18.11.19.
//

#ifndef SPN_COMPILER_V2_CODEGENOPERATIONS_H
#define SPN_COMPILER_V2_CODEGENOPERATIONS_H


#include <transform/BaseVisitor.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <unordered_map>

using namespace llvm;

typedef std::function<Value*(size_t index, IRBuilder<>& builder, LLVMContext& context)> InputVarValueMap;

class CodeGenOperations : public BaseVisitor {

public:

    CodeGenOperations(Module& m, Function& f, IRBuilder<>& b,
            InputVarValueMap inputMap);

    void visitInputvar(InputVar& n, arg_t arg) override ;

    void visitHistogram(Histogram& n, arg_t arg) override ;

    void visitProduct(Product& n, arg_t arg) override ;

    void visitSum(Sum& n, arg_t arg) override ;

    void visitWeightedSum(WeightedSum& n, arg_t arg) override ;

private:

    Module& module;

    Function& function;

    IRBuilder<>& builder;

    InputVarValueMap inputVarValueMap;

    std::unordered_map<GraphIRNode*, Value*> node2value;

    Type* getValueType();

    Value* getValueForNode(NodeReference node, arg_t arg);

};


#endif //SPN_COMPILER_V2_CODEGENOPERATIONS_H
