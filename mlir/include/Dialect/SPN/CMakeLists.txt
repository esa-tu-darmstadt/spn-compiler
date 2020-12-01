add_mlir_dialect(SPNOps spn)
add_mlir_doc(SPNDialect -gen-dialect-doc SPNDialect SPN/)
add_mlir_doc(SPNOps -gen-op-doc SPNOps SPN/)
add_mlir_doc(SPNPasses -gen-pass-doc SPNPasses SPN/)

set(LLVM_TARGET_DEFINITIONS SPNOps.td)
mlir_tablegen(SPNAttributes.h.inc -gen-struct-attr-decls)
mlir_tablegen(SPNAttributes.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRSPNAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS SPNOps.td)
mlir_tablegen(SPNEnums.h.inc -gen-enum-decls)
mlir_tablegen(SPNEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSPNEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS SPNPasses.td)
mlir_tablegen(SPNPasses.h.inc -gen-pass-decls -name SPN)
add_public_tablegen_target(MLIRSPNPassesIncGen)

set(LLVM_TARGET_DEFINITIONS SPNInterfaces.td)
mlir_tablegen(SPNInterfaces.h.inc -gen-op-interface-decls -name SPN)
mlir_tablegen(SPNInterfaces.cpp.inc -gen-op-interface-defs -name SPN)
add_public_tablegen_target(MLIRSPNInterfacesIncGen)