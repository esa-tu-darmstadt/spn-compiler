add_mlir_dialect(SPNOps spn)
add_mlir_doc(SPNDialect -gen-dialect-doc SPNDialect SPN/)
add_mlir_doc(SPNOps -gen-op-doc SPNOps SPN/)

set(LLVM_TARGET_DEFINITIONS SPNOps.td)
mlir_tablegen(SPNAttributes.h.inc -gen-struct-attr-decls)
mlir_tablegen(SPNAttributes.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRSPNAttributesIncGen)