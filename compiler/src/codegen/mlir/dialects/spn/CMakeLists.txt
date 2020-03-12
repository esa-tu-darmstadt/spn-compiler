include(MLIRTableGen.cmake)

mlir_tablegen(TARGET_NAME SPNOpInterfaceGen DEFINITION SPNOpInterfaces.td GENERATE_OP_INTERFACES)
mlir_tablegen(TARGET_NAME SPNOpsIncGen DEFINITION SPNOps.td GENERATE_STRUCT_ATTRIBUTES
        GENERATE_OPERATIONS DEPENDS ${SPNOpInterfaceGen_OP_INTERFACE_HEADER})

# The generation of the operations must depend on
# the generation of the interfaces.
add_dependencies(SPNOpsIncGen SPNOpInterfaceGen)

# Propagate important variables to parent scope.
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS} PARENT_SCOPE)