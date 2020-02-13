include(MLIRTableGen.cmake)

mlir_tablegen(SPNOpsIncGen SPNOps.td)

# Propagate important variables to parent scope.
set(TABLEGEN_HEADER ${TABLEGEN_HEADER} PARENT_SCOPE)
set(TABLEGEN_IMPL ${TABLEGEN_IMPL} PARENT_SCOPE)
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS} PARENT_SCOPE)