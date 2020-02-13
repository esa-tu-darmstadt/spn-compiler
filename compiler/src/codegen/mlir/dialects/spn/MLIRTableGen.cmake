function(mlir_tablegen target definition)

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

    get_filename_component(path ${definition} REALPATH)
    get_filename_component(name ${definition} NAME_WE)

    # Generate header containing all classes for the dialect using mlir-tblgen.
    set(${target}_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${name}.h.inc)
    add_custom_command(OUTPUT ${${target}_HEADER}
            COMMAND ${MLIR_TBLGEN} -gen-op-decls "-I=${MLIR_MAIN_SRC_DIR}" "-I=${MLIR_HEADER_DIR}" "-o=${${target}_HEADER}" ${path}
            DEPENDS "${path}")

    # Generate implementation of all classes for the dialect using mlir-tblgen.
    set(${target}_IMPL ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp.inc)
    add_custom_command(OUTPUT ${${target}_IMPL}
            COMMAND ${MLIR_TBLGEN} -gen-op-defs "-I=${MLIR_MAIN_SRC_DIR}" "-I=${MLIR_HEADER_DIR}" "-o=${${target}_IMPL}" ${path}
            DEPENDS "${path}")

    # Add custom target specifying the dependency on output of mlir-tblgen.
    add_custom_target(${target} DEPENDS ${TABLEGEN_HEADER} ${TABLEGEN_IMPL})

endfunction(mlir_tablegen)