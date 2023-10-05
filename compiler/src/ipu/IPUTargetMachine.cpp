#include "IPUTargetMachine.h"
#include "IPUTargetInfo.h"
#include "ipu/IPUSubtarget.h"
#include "llvm/Support/TargetRegistry.h"

using namespace spnc;
using namespace llvm;

void spnc::initializeIPUTarget() { llvm::RegisterTargetMachine<IPUTargetMachine> X(getTheIPUTarget()); }

static StringRef computeDataLayout(StringRef CPU) {
  if (CPU == "ipu1" || CPU == "ipu2" || CPU == "ipu21") {
    return "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:32-i128:64-f64:32-f128:"
           "64-v128:64-a:0:32-n32";
  } else
    report_fatal_error("Unknown CPU for IPU target: " + CPU);
}

static Reloc::Model getEffectiveRelocModel(const Triple &TT, Optional<Reloc::Model> RM) {
  if (!RM.hasValue())
    return Reloc::Static;
  return *RM;
}

IPUTargetMachine::IPUTargetMachine(const Target &T, const Triple &TT, StringRef CPU, StringRef FS,
                                   const TargetOptions &Options, Optional<Reloc::Model> RM,
                                   Optional<CodeModel::Model> CM, CodeGenOpt::Level OL, bool JIT)
    : LLVMTargetMachine(T, computeDataLayout(CPU), TT, CPU, FS, Options, getEffectiveRelocModel(TT, RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL), subtarget_(TT, CPU, "", FS, *this) {}

const llvm::TargetSubtargetInfo *IPUTargetMachine::getSubtargetImpl(const llvm::Function &) const {
  return &subtarget_;
}