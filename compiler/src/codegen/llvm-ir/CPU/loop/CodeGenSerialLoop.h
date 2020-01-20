//
// Created by lukas on 21.11.19.
//

#ifndef SPNC_CODEGENSERIALLOOP_H
#define SPNC_CODEGENSERIALLOOP_H

#include <codegen/llvm-ir/CPU/body/CodeGenBody.h>
#include "CodeGenLoop.h"

namespace spnc {

    class CodeGenSerialLoop : public CodeGenLoop {

    public:

        CodeGenSerialLoop(Module& m, IRGraph& g);

        void emitLoop(Function& function, IRBuilder<>& builder, Value* lowerBound, Value* upperBound) override;

        std::vector<Type*> constructInputArgumentTypes() override;

        std::vector<Type*> constructOutputArgumentTypes() override;

        static InputVarValueMap getDefaultInputMap(Function& f, IRBuilder<>& b, LLVMContext& c){
          return serialInputAccess{f, b, c};
        }

        static OutputAddressMap getDefaultOutputMap(Function& f, IRBuilder<>& b){
          return serialOutputAccess{f, b};
        }

    private:

        struct serialInputAccess {

            Function& function;

            IRBuilder<>& builder;

            LLVMContext& context;

            Value* operator()(size_t inputIndex, Value* indVar){
              auto paramBegin = function.arg_begin();
              auto inputArg = ++paramBegin;
              assert(inputArg->getType()->isPointerTy() && "Expecting input to be a pointer type!");
              assert(((PointerType*)inputArg->getType())->getElementType()->isAggregateType()
                     && "Expecting input to be a struct!");
              auto gep = builder.CreateGEP(inputArg,
                                           {indVar,
                                            ConstantInt::get(IntegerType::get(context, 32), inputIndex)}, "gep_input");
              return builder.CreateLoad(gep, "input_value");
            }

        };

        struct serialOutputAccess {

            Function& function;

            IRBuilder<>& builder;

            Value* operator()(Value* indVar){
              auto paramBegin = function.arg_begin();
              paramBegin++;
              auto outputPtr = ++paramBegin;
              return builder.CreateGEP(outputPtr, indVar, "store.address");
            }
        };



    };
}

#endif //SPNC_CODEGENSERIALLOOP_H
