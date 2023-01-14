#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "option/Options.h"
#include "Kernel.h"
#include "llvm/Target/TargetMachine.h"


namespace spnc {

// The CIRCT toolchain deviates quite strongly from the vanilla MLIR toolchain mostly in terms
// of result file production and external library linking.

class HDLSources {
public:
  
};

class CIRCTToolchain {
public:
  static void initializeMLIRContext(mlir::MLIRContext& ctx);
};

}