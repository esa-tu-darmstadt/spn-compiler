//
// Created by mhalk on 2/6/20.
//

#include <algorithm>
#include <iostream>
#include "NumericalValueTracingPass.h"

using namespace llvm;

char NumericalValueTracingPass::ID = 43;

static RegisterPass<NumericalValueTracingPass>
        NumericalValueTracingPassReg("numvaltrace",
                "Traces SPN-Nodes' instruction-values, e.g. for numerical analysis.",
                false /* Only looks at CFG */, false /* Analysis Pass */);

bool NumericalValueTracingPass::doInitialization(Module &MOD) {
  M = &MOD;
  auto &CTX = M->getContext(); // CTX holds global information
  Builder = new IRBuilder<>(CTX);
  MDKindID = CTX.getMDKindID(spnc::TraceMDName);
  return false; // Module has not been modified
}

bool NumericalValueTracingPass::runOnBasicBlock(BasicBlock &BB) {
  bool Traced = false;
  resetTracedInstructions();
  collectTracedInstructions(BB);

  for (auto tag : tracedTags) {
    Traced |= traceInstructions(tracedInstructions.find(tag)->second);
  }

  return Traced;
}

void NumericalValueTracingPass::resetTracedInstructions() {
  // Clear all collected data and prepare for new instructions.
  tracedInstructions.clear();
  for (auto tag : tracedTags) {
    tracedInstructions[tag] = std::vector<Instruction*>();
  }
}

void NumericalValueTracingPass::collectTracedInstructions(BasicBlock &BB) {
  for (Instruction &I: BB) {
    // Only specific Instructions are "interesting"
    if (auto MD = I.getMetadata(MDKindID)) {
      // Extract metadata which was stored as a constant int, based on the spnc::TraceMDTag enum.
      Constant* val = dyn_cast<ConstantAsMetadata>(MD->getOperand(0))->getValue();
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

  // Mini Trace Report
  std::cout << "Traced: '" << std::string(BB.getName()) << "'" << std::endl;
  std::cout << "\t Sum-Insn:         " <<
            tracedInstructions.find(spnc::TraceMDTag::Sum)->second.size() << std::endl;
  std::cout << "\t WeightedSum-Insn: " <<
            tracedInstructions.find(spnc::TraceMDTag::WeightedSum)->second.size() << std::endl;
  std::cout << "\t Product-Insn:     " <<
            tracedInstructions.find(spnc::TraceMDTag::Product)->second.size() << std::endl;
  std::cout << "\t Histogram-Insn:   " <<
            tracedInstructions.find(spnc::TraceMDTag::Histogram)->second.size() << std::endl;

}

bool NumericalValueTracingPass::traceInstructions(const std::vector<Instruction*>& INSN) {
  bool TracedInstruction = false;
  for (Instruction* I : INSN) {
    Builder->SetInsertPoint(I->getNextNode());
    createCallTrace(I);
    // An instruction / value  was traced; i.e.: BB modified
    TracedInstruction = true;
  }

  return TracedInstruction;
}

void NumericalValueTracingPass::createCallTrace(Value* value) {
  // ToDo: Review _TRACE_ function, esp. its signature & additionally possible traced instruction values. (2020-FEB-08)
  const std::string Name = "_TRACE_";
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
