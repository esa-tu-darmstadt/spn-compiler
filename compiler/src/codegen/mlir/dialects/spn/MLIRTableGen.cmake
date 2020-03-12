function(mlir_tablegen)
    set(options GENERATE_OPERATIONS GENERATE_STRUCT_ATTRIBUTES GENERATE_OP_INTERFACES)
    set(oneValueArgs TARGET_NAME DEFINITION)
    set(multiValueArgs DEPENDS)
    cmake_parse_arguments(TBLGEN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT TBLGEN_TARGET_NAME)
        message(FATAL_ERROR "You must provide a target name!")
    endif (NOT TBLGEN_TARGET_NAME)

    if (NOT TBLGEN_DEFINITION)
        message(FATAL_ERROR "You must provide a definition file!")
    endif (NOT TBLGEN_DEFINITION)

    # Check for directory containing all LLVM tool binaries.
    if (NOT DEFINED LLVM_TOOLS_BINARY_DIR)
        message(FATAL_ERROR "Expecting definition of LLVM_TOOLS_BINARY_DIR!")
    endif (NOT DEFINED LLVM_TOOLS_BINARY_DIR)

    # Check for directory containing LLVM source tree.
    if (NOT DEFINED LLVM_BUILD_MAIN_SRC_DIR)
        message(FATAL_ERROR "Expecting definition of LLVM_BUILD_MAIN_SRC_DIR!")
    endif (NOT DEFINED LLVM_BUILD_MAIN_SRC_DIR)

    # Add LLVM tool binary directory to path.
    LIST(APPEND CMAKE_PROGRAM_PATH "${LLVM_TOOLS_BINARY_DIR}")

    # Try to find mlir-tblgen executable, prefereable in LLVM_TOOLS_BINARY_DIR.
    find_program(MLIR_TBLGEN mlir-tblgen "${LLVM_TOOLS_BINARY_DIR}")
    if (NOT MLIR_TBLGEN)
        message(FATAL_ERROR "Did not find mlir-tblgen executable!")
    endif (NOT MLIR_TBLGEN)
    message(STATUS "Using mlir-tblgen: ${MLIR_TBLGEN}")

    # Find directory containing MLIR project headers in LLVM source tree.
    set(MLIR_HEADER_DIR ${LLVM_BUILD_MAIN_SRC_DIR}/../mlir/include)
    if (NOT EXISTS ${MLIR_HEADER_DIR})
        message(FATAL_ERROR "Did not find MLIR headers in ${MLIR_HEADER_DIR}!")
    endif (NOT EXISTS ${MLIR_HEADER_DIR})

    # Some MLIR headers are generated from table-gen in LLVM build directory.
    set(MLIR_GEN_HEADER_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
    if (NOT EXISTS ${MLIR_GEN_HEADER_DIR})
        message(FATAL_ERROR "Did not find generated MLIR headers in ${MLIR_GEN_HEADER_DIR}!")
    endif (NOT EXISTS ${MLIR_GEN_HEADER_DIR})

    # Include both, the MLIR project headers and the generated headers for MLIR.
    set(MLIR_INCLUDE_DIRS ${MLIR_HEADER_DIR} ";" ${MLIR_GEN_HEADER_DIR} PARENT_SCOPE)

    # Find MLIR sources in LLVM source tree, required for MLIR table-gen.
    set(MLIR_MAIN_SRC_DIR ${LLVM_BUILD_MAIN_SRC_DIR}/../mlir/lib)
    if (NOT EXISTS ${MLIR_MAIN_SRC_DIR})
        message(FATAL_ERROR "Did not find MLIR sources in ${MLIR_MAIN_SRC_DIR}!")
    endif (NOT EXISTS ${MLIR_MAIN_SRC_DIR})

    get_filename_component(path ${TBLGEN_DEFINITION} REALPATH)
    get_filename_component(name ${TBLGEN_DEFINITION} NAME_WE)

    if (${TBLGEN_GENERATE_OPERATIONS})
        # Generate header containing all classes for the dialect operations using mlir-tblgen.
        set(${TBLGEN_TARGET_NAME}_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${name}.h.inc)
        add_custom_command(OUTPUT ${${TBLGEN_TARGET_NAME}_HEADER}
                COMMAND ${MLIR_TBLGEN} -gen-op-decls "-I=${MLIR_MAIN_SRC_DIR}" "-I=${MLIR_HEADER_DIR}"
                "-I=${CMAKE_CURRENT_SOURCE_DIR}" "-o=${${TBLGEN_TARGET_NAME}_HEADER}" ${path}
                DEPENDS "${path}" "${TBLGEN_DEPENDS}")


        # Generate implementation of all classes for the dialect operations using mlir-tblgen.
        set(${TBLGEN_TARGET_NAME}_IMPL ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp.inc)
        add_custom_command(OUTPUT ${${TBLGEN_TARGET_NAME}_IMPL}
                COMMAND ${MLIR_TBLGEN} -gen-op-defs "-I=${MLIR_MAIN_SRC_DIR}" "-I=${MLIR_HEADER_DIR}"
                "-I=${CMAKE_CURRENT_SOURCE_DIR}" "-o=${${TBLGEN_TARGET_NAME}_IMPL}" ${path}
                DEPENDS "${path}" "${TBLGEN_DEPENDS}")
    endif (${TBLGEN_GENERATE_OPERATIONS})


    if (${TBLGEN_GENERATE_STRUCT_ATTRIBUTES})
        # Generate header containing all classes for the dialect struct attributes using mlir-tblgen.
        set(${TBLGEN_TARGET_NAME}_ATTR_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${name}.attr.h.inc)
        add_custom_command(OUTPUT ${${TBLGEN_TARGET_NAME}_ATTR_HEADER}
                COMMAND ${MLIR_TBLGEN} -gen-struct-attr-decls "-I=${MLIR_MAIN_SRC_DIR}" "-I=${MLIR_HEADER_DIR}"
                "-I=${CMAKE_CURRENT_SOURCE_DIR}" "-o=${${TBLGEN_TARGET_NAME}_ATTR_HEADER}" ${path}
                DEPENDS "${path}" "${TBLGEN_DEPENDS}")

        # Generate implementation of all classes for the dialect struct attributes using mlir-tblgen.
        set(${TBLGEN_TARGET_NAME}_ATTR_IMPL ${CMAKE_CURRENT_BINARY_DIR}/${name}.attr.cpp.inc)
        add_custom_command(OUTPUT ${${TBLGEN_TARGET_NAME}_ATTR_IMPL}
                COMMAND ${MLIR_TBLGEN} -gen-struct-attr-defs "-I=${MLIR_MAIN_SRC_DIR}" "-I=${MLIR_HEADER_DIR}"
                "-I=${CMAKE_CURRENT_SOURCE_DIR}" "-o=${${TBLGEN_TARGET_NAME}_ATTR_IMPL}" ${path}
                DEPENDS "${path}" "${TBLGEN_DEPENDS}")
    endif (${TBLGEN_GENERATE_STRUCT_ATTRIBUTES})

    if (${TBLGEN_GENERATE_OP_INTERFACES})
        # Generate header containing all classes for the operation interfaces using mlir-tblgen.
        set(${TBLGEN_TARGET_NAME}_OP_INTERFACE_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${name}.op.interface.h.inc)
        add_custom_command(OUTPUT ${${TBLGEN_TARGET_NAME}_OP_INTERFACE_HEADER}
                COMMAND ${MLIR_TBLGEN} -gen-op-interface-decls "-I=${MLIR_MAIN_SRC_DIR}" "-I=${MLIR_HEADER_DIR}"
                "-I=${CMAKE_CURRENT_SOURCE_DIR}" "-o=${${TBLGEN_TARGET_NAME}_OP_INTERFACE_HEADER}" ${path}
                DEPENDS "${path}" "${TBLGEN_DEPENDS}")

        # Generate implementation of all classes for the operation interfaces using mlir-tblgen.
        set(${TBLGEN_TARGET_NAME}_OP_INTERFACE_IMPL ${CMAKE_CURRENT_BINARY_DIR}/${name}.op.interface.cpp.inc)
        add_custom_command(OUTPUT ${${TBLGEN_TARGET_NAME}_OP_INTERFACE_IMPL}
                COMMAND ${MLIR_TBLGEN} -gen-op-interface-defs "-I=${MLIR_MAIN_SRC_DIR}" "-I=${MLIR_HEADER_DIR}"
                "-I=${CMAKE_CURRENT_SOURCE_DIR}" "-o=${${TBLGEN_TARGET_NAME}_OP_INTERFACE_IMPL}" ${path}
                DEPENDS "${path}" "${TBLGEN_DEPENDS}")

    endif (${TBLGEN_GENERATE_OP_INTERFACES})
    set(${TBLGEN_TARGET_NAME}_OP_INTERFACE_HEADER ${${TBLGEN_TARGET_NAME}_OP_INTERFACE_HEADER} PARENT_SCOPE)

    # Add custom target specifying the dependency on output of mlir-tblgen.
    add_custom_target(${TBLGEN_TARGET_NAME}
            DEPENDS ${${TBLGEN_TARGET_NAME}_HEADER}
            ${${TBLGEN_TARGET_NAME}_ATTR_HEADER}
            ${${TBLGEN_TARGET_NAME}_OP_INTERFACE_HEADER}
            ${${TBLGEN_TARGET_NAME}_IMPL}
            ${${TBLGEN_TARGET_NAME}_ATTR_IMPL}
            ${${TBLGEN_TARGET_NAME}_OP_INTERFACE_IMPL}
            ${TBLGEN_DEPENDS})

endfunction(mlir_tablegen)