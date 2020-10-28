#ifndef MLIR_DIALECT_SPN_PASSES
#define MLIR_DIALECT_SPN_PASSES

include "mlir/Pass/PassBase.td"

def SPNOpSimplifier : Pass<"spn-op-simplifier", "ModuleOp"> {
  let summary = "Simplify SPN operations";
  let constructor = "mlir::spn::createSPNOpSimplifierPass()";
}

#endif // MLIR_DIALECT_SPN_PASSES