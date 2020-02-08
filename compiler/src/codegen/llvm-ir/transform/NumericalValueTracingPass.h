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
  * Header file for the LLVM loadable plugin to profile computations / numerical values
  */

/*
  The NumericalValueTracingPass will iterate over Functions and their BasicBlocks.
  Basically: information is collected, then the respective "Inst" is profiled.
  Only the actually used library-functions will be declared in the process!
  (Therefore, fail on test: "ut_memls_declare_interface")
  If everything is set-up properly "memTrace.out" will be written on execution
  of a resulting binary. (Obviously requires the "memprofile/trace" libraries)
  Example: make && ./memls++ . ./test/getstarted/simplefunc.cpp && ./a.out
*/

using namespace llvm;

class NumericalValueTracingPass : public BasicBlockPass {
public:
  // Register our pass
  static char ID;

  NumericalValueTracingPass() : BasicBlockPass(ID) {}

  bool doInitialization(Module &M) override;

  bool runOnBasicBlock(BasicBlock &BB) override;

  void collectTracedInstructions(BasicBlock &BB);

  bool traceInstructions(const std::vector<Instruction*>& I);

  void createCallTrace(Value* value);

  void resetTracedInstructions();

private:
  IRBuilder<> *Builder{};
  Module *M{};
  std::map<spnc::MetadataTag, std::vector<Instruction*>> tracedInstructions;

  std::vector<spnc::MetadataTag> tracedTags = {spnc::MetadataTag::Sum, spnc::MetadataTag::WeightedSum,
                                               spnc::MetadataTag::Product, spnc::MetadataTag::Histogram };
};

#endif //SPNC_NUMERICALVALUETRACINGPASS_H
