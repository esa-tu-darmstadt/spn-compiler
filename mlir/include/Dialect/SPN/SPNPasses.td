#ifndef MLIR_DIALECT_SPN_PASSES
#define MLIR_DIALECT_SPN_PASSES

include "mlir/Pass/PassBase.td"

def SPNOpSimplifier : Pass<"spn-op-simplifier", "ModuleOp"> {
  let summary = "Simplify SPN operations";
  let constructor = "mlir::spn::createSPNOpSimplifierPass()";
}

def SPNTypePinning : Pass<"spn-type-pinning", "ModuleOp"> {
  let summary = "Pin actual type for probability values";
  let constructor = "mlir::spn::createSPNTypePinningPass()";
}

def SPNVectorization : Pass<"spn-vectorization", "ModuleOp"> {
  let summary = "Vectorize SPN execution";
  let constructor = "mlir::spn::createSPNVectorizationPass()";
}

#endif // MLIR_DIALECT_SPN_PASSES