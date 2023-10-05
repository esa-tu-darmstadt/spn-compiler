#include "IPUTargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

using namespace spnc;

llvm::Target &spnc::getTheIPUTarget() {
    static llvm::Target TheIPUTarget;
    return TheIPUTarget;
}

void spnc::initializeIPUTargetInfo() {
  llvm::RegisterTarget<llvm::Triple::UnknownArch> X(getTheIPUTarget(), "colossus",
                                    "GraphCore Colossus", "");
}