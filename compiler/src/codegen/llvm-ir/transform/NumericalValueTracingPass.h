//
// Created by mhalk on 2/6/20.
//

#ifndef SPNC_NUMERICALVALUETRACINGPASS_H
#define SPNC_NUMERICALVALUETRACINGPASS_H

#include <llvm/Pass.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <codegen/llvm-ir/CPU/body/CodeGenBody.h>

/**
  * Header file for the LLVM loadable plugin to trace computations / numerical values
  */

/*
  The NumericalValueTracingPass will iterate over BasicBlocks.
  Basically: information is collected, then the respective "Inst" is traced.
*/

using namespace llvm;

class NumericalValueTracingPass : public BasicBlockPass {
public:
  // Register our pass
  static char ID;

  NumericalValueTracingPass() : BasicBlockPass(ID) {}

  bool doInitialization(Module &M) override;

  bool runOnBasicBlock(BasicBlock &BB) override;

private:

  void collectTracedInstructions(BasicBlock &BB);

  bool traceInstructions(const std::vector<Instruction*>& I);

  void createCallTrace(Value* value);

  void resetTracedInstructions();

  IRBuilder<> *Builder{};
  Module *M{};
  ushort MDKindID=std::numeric_limits<ushort>::max();

  std::map<spnc::TraceMDTag, std::vector<Instruction*>> tracedInstructions;

  std::vector<spnc::TraceMDTag> tracedTags = {spnc::TraceMDTag::Sum, spnc::TraceMDTag::WeightedSum,
                                              spnc::TraceMDTag::Product, spnc::TraceMDTag::Histogram};
};

#endif //SPNC_NUMERICALVALUETRACINGPASS_H
