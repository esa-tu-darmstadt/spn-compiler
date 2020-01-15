//
// Created by ls on 10/9/19.
//

#include "LLVMCodegen.h"
#include "codegen/llvm-ir/IREmitter.h"
#include "codegen/shared/PackingSolver.h"
#include "codegen/shared/PackingTrivial.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <queue>

#define SIMD_WIDTH 4
#define USE_SOLVER 1

LLVMCodegen::LLVMCodegen() : builder{context} {
    module = std::make_unique<Module>("spn-llvm", context);
}

void LLVMCodegen::generateLLVMIR(IRGraph &graph, bool vectorize) {
    auto intType = Type::getInt32Ty(context);
    std::vector<Type*> argTypes{PointerType::get(intType, 0), Type::getDoublePtrTy(context, 0), Type::getInt64Ty(context)};
    auto functionType = FunctionType::get(Type::getVoidTy(context), argTypes, false);
    func = Function::Create(functionType, Function::ExternalLinkage, "spn_element", module.get());
    auto bb = BasicBlock::Create(context, "main", func);
    auto olh = BasicBlock::Create(context, "outer-loop-header", func);
    auto ilh = BasicBlock::Create(context, "inner-loop-header", func);
    auto ilb = BasicBlock::Create(context, "inner-loop-body", func);
    auto ile = BasicBlock::Create(context, "inner-loop-exit", func);
    auto ex = BasicBlock::Create(context, "exit", func);
    
    auto iterations = ConstantInt::getSigned(Type::getInt64Ty(context), 1);
    auto arg_it = func->arg_begin();
    auto input = arg_it++;
    auto output = arg_it++;
    auto count = arg_it;

    builder.SetInsertPoint(bb);
    builder.CreateBr(olh);

    builder.SetInsertPoint(olh);
    auto cur_iteration = builder.CreatePHI(Type::getInt64Ty(context), 2);
    cur_iteration->addIncoming(ConstantInt::getSigned(Type::getInt64Ty(context), 0), bb);
    auto cmpo = builder.CreateICmpSLT(cur_iteration, iterations);
    builder.CreateCondBr(cmpo, ilh, ex);

    builder.SetInsertPoint(ilh);
    auto cur_count = builder.CreatePHI(Type::getInt64Ty(context), 2);
    cur_count->addIncoming(ConstantInt::getSigned(Type::getInt64Ty(context), 0), olh);
    auto cmpi = builder.CreateICmpSLT(cur_count, count);
    builder.CreateCondBr(cmpi, ilb, ile);

    builder.SetInsertPoint(ilb);
    auto in_offset = builder.CreateMul(cur_count, ConstantInt::getSigned(Type::getInt64Ty(context), graph.inputs->size()));
    auto in_ptr = builder.CreateGEP(input, {in_offset});
    auto out_ptr = builder.CreateGEP(output, {cur_count});

    std::unordered_map<std::string, size_t> partOf;
    std::unordered_map<size_t, std::unordered_set<size_t>> directVecInputs;
    std::vector<std::vector<NodeReference>> vectors;
    if (vectorize) {
      if (USE_SOLVER) {
        PackingSolver packer;
        auto vecInfo = packer.getVectorization(graph, SIMD_WIDTH);
	partOf = vecInfo.partOf;
	directVecInputs = vecInfo.directVecInputs;
	vectors = vecInfo.vectors;
      } else {
	PackingTrivial packer;
        auto vecInfo = packer.getVectorization(graph, SIMD_WIDTH);
	partOf = vecInfo.partOf;
	directVecInputs = vecInfo.directVecInputs;
	vectors = vecInfo.vectors;
      }
    }

    IREmitter codeEmitter(partOf, directVecInputs, vectors, in_ptr, func,
                          context, builder, module.get(), SIMD_WIDTH);
    
    graph.rootNode->accept(codeEmitter, {});
    builder.CreateStore(codeEmitter.getNodeMap()[graph.rootNode->id()].val,
                        out_ptr, true);
    
    auto inci = builder.CreateAdd(cur_count, ConstantInt::getSigned(Type::getInt64Ty(context), 1));
    cur_count->addIncoming(inci, builder.GetInsertBlock());
    builder.CreateBr(ilh);

    builder.SetInsertPoint(ile);
    auto inco = builder.CreateAdd(cur_iteration, ConstantInt::getSigned(Type::getInt64Ty(context), 1));
    cur_iteration->addIncoming(inco, ile);
    builder.CreateBr(olh);

    builder.SetInsertPoint(ex);
    builder.CreateRetVoid();
    
    std::error_code EC;
    llvm::raw_fd_ostream OS("/Users/johannesschulte/Desktop/Uni/MT/cpp-spn-compiler/debLLVMBuild/out.bc", EC);
    //module->print(llvm::errs(), nullptr);
    WriteBitcodeToFile(*module, OS);
}
