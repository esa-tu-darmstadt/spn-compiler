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


size_t CostInfo::histogramCost(std::multiset<size_t, std::greater<size_t>> inputHistos) {
  if (AVX2) {
    auto it = inputHistos.begin();
    // The first two lanes can be calculated with no extra insertion cost, since xmm is 128 bit wide
    if (*it == 0)
      return 0;
    size_t cost = (*it-1)*scalarArithCost;
    if (inputHistos.size() > 2) {
      // This is an overapproximation: in the optimal case, the two lanes with
      // the most histogram children will be 1 and 2 or 3 and 4, i.e. for the
      // other two lanes, only as many arithmetic operations as the third most
      // histogram children -1 will be needed. But there can never be more than
      // first #histo-1 + second +histo-1 arithmetic operations necessary
      it++;
      if (*it == 0)
	return cost;
      cost += (*it-1)*scalarArithCost;
      cost += insertCost;
    }
    return cost;
  }

  size_t cost = 0;
  for (auto& lane : inputHistos) {
    // This is a very conservative estimate, on most architectures we do not
    // need insert operations for all lanes
    cost += (lane-1)*scalarArithCost+insertCost;
  }
}
size_t CostInfo::getHistogramPenalty(size_t width) {
  if (AVX2) {
    if (width > 2)
      return insertCost;
    return 0;
  }
  return width*insertCost;
}

float CostInfo::getExtractCost(size_t width) {
  if (AVX2) {
    // Since we don't know which lane the value to be extracted will be in, we
    // return the average here
    // In general, the first lane can extracted without cost, the second lane
    // requires one vpermilpd, the third one vextractf128, and the fourth one
    // an additional vpermilpd using the vextractf128 result
    switch (width) {
    case 2:
      return float(extractCost) / 2.0;
    case 3:
      return float(extractCost) * 2.0 / 3.0;
    case 4:
      return float(extractCost) * 3.0 / 4.0;
    default:
      assert(false);
    }
  }
  return float(extractCost);
}

float CostInfo::getInsertCost(size_t width) {
  if (AVX2) {
    // Since we don't know which lane the value to be inserted will be in, we
    // return the average here
    // In general, the first lane doesn't need an insert, the second lane
    // requires one vpermilpd, the third one vinsertf128, and the fourth one
    // an additional vpermilpd to produce the input for vinsertf128
    switch (width) {
    case 2:
      return float(insertCost) / 2.0;
    case 3:
      return float(insertCost) * 2.0 / 3.0;
    case 4:
      return float(insertCost) * 3.0 / 4.0;
    default:
      assert(false);
    }
  }
  return insertCost;
}
