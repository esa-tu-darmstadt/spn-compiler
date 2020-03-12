//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_CODEGENSERIALLOOP_H
#define SPNC_CODEGENSERIALLOOP_H

#include <codegen/llvm-ir/CPU/body/CodeGenBody.h>
#include <graph-ir/IRGraph.h>
#include "CodeGenLoop.h"

namespace spnc {

  ///
  /// Code generation for a serial (i.e., non-vectorized) loop.
  class CodeGenSerialLoop : public CodeGenLoop {

  public:

    using CodeGenLoop::CodeGenLoop;

    void emitLoop(Function& function, IRBuilder<>& builder, Value* lowerBound, Value* upperBound) override;

    std::vector<Type*> constructInputArgumentTypes() override;

    std::vector<Type*> constructOutputArgumentTypes() override;

    /// Get the default input-to-LLVM-value mapping for the serial loop. This corresponds to a
    /// a simple extraction from the input struct.
    /// \param f LLVM IR function.
    /// \param b IRBuilder used to create LLVM IR instructions.
    /// \return Mapping from SPN feature to LLVM IR value.
    static InputVarValueMap getDefaultInputMap(Function& f, IRBuilder<>& b) {
      return serialInputAccess{f, b};
    }

    /// Get the default mapping of query index to output store location.
    /// \param f LLVM IR function.
    /// \param b IRBuilder used to create LLVM IR instructions.
    /// \return Mapping from SPN query index to output store location address.
    static OutputAddressMap getDefaultOutputMap(Function& f, IRBuilder<>& b) {
      return serialOutputAccess{f, b};
    }

  private:

    struct serialInputAccess {

      Function& function;

      IRBuilder<>& builder;

      Value* operator()(size_t inputIndex, Value* indVar) {
        auto paramBegin = function.arg_begin();
        auto inputArg = ++paramBegin;
        assert(inputArg->getType()->isPointerTy() && "Expecting input to be a pointer type!");
        assert(((PointerType*) inputArg->getType())->getElementType()->isAggregateType()
                   && "Expecting input to be a struct!");
        // Create a load reading the struct element at inputIndex of the struct at indVar.
        auto gep = builder.CreateGEP(inputArg,
                                     {indVar,
                                      ConstantInt::get(IntegerType::get(builder.getContext(), 32),
                                                       inputIndex)}, "gep_input");
        return builder.CreateLoad(gep, "input_value");
      }

    };

    struct serialOutputAccess {

      Function& function;

      IRBuilder<>& builder;

      Value* operator()(Value* indVar) {
        auto paramBegin = function.arg_begin();
        paramBegin++;
        auto outputPtr = ++paramBegin;
        return builder.CreateGEP(outputPtr, indVar, "store.address");
      }
    };

  };
}

#endif //SPNC_CODEGENSERIALLOOP_H
