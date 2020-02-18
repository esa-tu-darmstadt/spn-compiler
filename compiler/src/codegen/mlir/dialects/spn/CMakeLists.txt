include(MLIRTableGen.cmake)

mlir_tablegen(TARGET_NAME SPNOpsIncGen DEFINITION SPNOps.td GENERATE_STRUCT_ATTRIBUTES)

# Propagate important variables to parent scope.
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS} PARENT_SCOPE)