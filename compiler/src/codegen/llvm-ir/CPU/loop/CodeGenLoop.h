//
// Created by lukas on 20.11.19.
//

#ifndef SPNC_CODEGENLOOP_H
#define SPNC_CODEGENLOOP_H

#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <graph-ir/GraphIRNode.h>

using namespace llvm;

class CodeGenLoop {
public:
    CodeGenLoop(Module& m, IRGraph& g) : module{m}, graph{g} {}

    virtual void emitLoop(Function& function, IRBuilder<>& builder, Value* lowerBound, Value* upperBound) = 0;

    virtual std::vector<Type*> constructInputArgumentTypes() = 0;

    virtual std::vector<Type*> constructOutputArgumentTypes() = 0;

protected:
    Module& module;
    IRGraph& graph;
};

#endif //SPNC_CODEGENLOOP_H
