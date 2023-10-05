#pragma once

namespace llvm {
class Target;
}

namespace spnc {
llvm::Target &getTheIPUTarget();
void initializeIPUTargetInfo();
}