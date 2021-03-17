#ifndef MLIR_DIALECT_LoSPN_PASSES
#define MLIR_DIALECT_LoSPN_PASSES

include "mlir/Pass/PassBase.td"

def LoSPNBufferize : Pass<"lospn-bufferize", "ModuleOp"> {
    let summary = "Bufferize LoSPN operations";
    let constructor = "mlir::spn::low::createLoSPNBufferizePass()";
}

#endif // MLIR_DIALECT_LoSPN_PASSES