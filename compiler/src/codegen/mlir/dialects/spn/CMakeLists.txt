include(MLIRTableGen.cmake)

mlir_tablegen(SPNOpsIncGen SPNOps.td)

# Propagate important variables to parent scope.
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS} PARENT_SCOPE)