#include "CostInfo.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetOptions.h"


#include "llvm/Target/TargetMachine.h"
#include <iostream>

using namespace llvm;


CostInfo::CostInfo(size_t width) {
  if (ARCH == AVX2) {
    raw_os_ostream os{std::cout};
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    std::string triple = "x86_64-apple-darwin";
    std::string err;
    auto target = TargetRegistry::lookupTarget(triple, err);
    TargetOptions to;
    auto tm = target->createTargetMachine(triple, "skylake", "+fast-gather", to, None);

    LLVMContext context;
    
    auto module = std::make_unique<Module>("tti-stub-module", context);
    
    auto functionType = FunctionType::get(Type::getVoidTy(context), {}, false);
    auto func = Function::Create(functionType, Function::ExternalLinkage, "spn_element", module.get());
    auto tti = tm->getTargetTransformInfo(*func);
    scalarArithCost = tti.getArithmeticInstrCost(Instruction::FMul, Type::getDoubleTy(context));
    vecArithCost = tti.getArithmeticInstrCost(Instruction::FMul, VectorType::get(Type::getDoubleTy(context), width));
    insertCost =
        tti.getVectorInstrCost(Instruction::InsertElement,
                               VectorType::get(Type::getDoubleTy(context), width));
    extractCost = tti.getVectorInstrCost(
        Instruction::ExtractElement,
        VectorType::get(Type::getDoubleTy(context), width));
    return;
  }
  assert(false);

}
