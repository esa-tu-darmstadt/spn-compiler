//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <iostream>
#include "NumericalValueTracingPass.h"

using namespace llvm;

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "NumericalValueTracingPass", "v0.1",
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef PassName, ModulePassManager &MPM, ...) {
          if(PassName == "spn-numerical-trace-pass"){
            MPM.addPass(NumericalValueTracingPass());
            return true;
          }
          return false;
        }
      );
    }
  };
}

PreservedAnalyses NumericalValueTracingPass::run(Module &MOD, ModuleAnalysisManager &MAM) {
  // Initialize class members
  M = &MOD;
  auto &CTX = M->getContext();
  Builder = new IRBuilder<>(CTX);
  MDKindID = CTX.getMDKindID(spnc::TraceMDName);

  // Run the pass on all Functions
  for (Function &F : MOD) {
    run(F);
  }

  // Previous analysis results *should* still apply
  return PreservedAnalyses::all();
}

void NumericalValueTracingPass::run(Function &F) {
  resetTracedInstructions();
  collectTracedInstructions(F);

  for (auto tag : tracedTags) {
    std::vector<Instruction*> TI = tracedInstructions.find(tag)->second;
    if (!TI.empty()) {
      traced = true;
      traceInstructions(TI);
    }
  }
}

void NumericalValueTracingPass::resetTracedInstructions() {
  // Clear all collected data and prepare for new instructions.
  tracedInstructions.clear();
  for (auto tag : tracedTags) {
    tracedInstructions[tag] = std::vector<Instruction*>();
  }
}

void NumericalValueTracingPass::collectTracedInstructions(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &I: BB) {
      // Only specific Instructions are "interesting"
      if (auto MD = I.getMetadata(MDKindID)) {
        // Extract metadata which was stored as a constant int, based on the spnc::TraceMDTag enum.
        Constant *val = dyn_cast<ConstantAsMetadata>(MD->getOperand(0))->getValue();
        int64_t metadata = cast<ConstantInt>(val)->getSExtValue();

        // Convert the collected data back to a spnc::TraceMDTag.
        auto tag = spnc::TraceMDTag(metadata);
        switch (tag) {
          case spnc::TraceMDTag::Sum :
          case spnc::TraceMDTag::WeightedSum :
          case spnc::TraceMDTag::Product :
          case spnc::TraceMDTag::Histogram :
            tracedInstructions.find(tag)->second.push_back(&I);
            break;
          default:
            errs() << "Encountered unknown Metadata-ID '" << metadata << "'.";
            assert(false);
        }
      }
    }
  }

  // Mini Trace Report
  std::cout << "Traced: '" << std::string(F.getName()) << "'" << std::endl;
  std::cout << "\t Sum-Insn:         " <<
            tracedInstructions.find(spnc::TraceMDTag::Sum)->second.size() << std::endl;
  std::cout << "\t WeightedSum-Insn: " <<
            tracedInstructions.find(spnc::TraceMDTag::WeightedSum)->second.size() << std::endl;
  std::cout << "\t Product-Insn:     " <<
            tracedInstructions.find(spnc::TraceMDTag::Product)->second.size() << std::endl;
  std::cout << "\t Histogram-Insn:   " <<
            tracedInstructions.find(spnc::TraceMDTag::Histogram)->second.size() << std::endl;

}

void NumericalValueTracingPass::traceInstructions(const std::vector<Instruction*>& INSN) {
  for (Instruction* I : INSN) {
    Builder->SetInsertPoint(I->getNextNode());
    createCallTrace(I);
  }
}

void NumericalValueTracingPass::createCallTrace(Value* value) {
  // ToDo: Review _TRACE_ function, esp. its signature & additionally possible traced instruction values. (2020-FEB-08)
  const std::string Name = "trace";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = { Builder->getDoubleTy() };

    FunctionType *Ty = FunctionType::get(Builder->getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = { value };

  Builder->CreateCall(F, Args);
}
