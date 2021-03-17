add_mlir_dialect(LoSPNOps lo_spn)
add_mlir_doc(LoSPNDialect -gen-dialect-doc LoSPNDialect LoSPN/)
add_mlir_doc(LoSPNOps -gen-op-doc LoSPNOps LoSPN/)

set(LLVM_TARGET_DEFINITIONS LoSPNOps.td)
mlir_tablegen(LoSPNAttributes.h.inc -gen-struct-attr-decls)
mlir_tablegen(LoSPNAttributes.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRLoSPNAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS LoSPNInterfaces.td)
mlir_tablegen(LoSPNInterfaces.h.inc -gen-op-interface-decls -name LoSPN)
mlir_tablegen(LoSPNInterfaces.cpp.inc -gen-op-interface-defs -name LoSPN)
add_public_tablegen_target(MLIRLoSPNInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS LoSPNPasses.td)
mlir_tablegen(LoSPNPasses.h.inc -gen-pass-decls -name LoSPN)
add_public_tablegen_target(MLIRLoSPNPassesIncGen)