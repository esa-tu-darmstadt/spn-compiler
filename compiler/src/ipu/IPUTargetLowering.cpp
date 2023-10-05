#include "IPUTargetLowering.h"

using namespace spnc;
using namespace llvm;

IPUTargetLowering::IPUTargetLowering(const TargetMachine &TM, const IPUSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {}