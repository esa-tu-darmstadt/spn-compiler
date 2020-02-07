include(${LLVM_CMAKE_DIR}/TableGen.cmake)
LIST(APPEND CMAKE_PROGRAM_PATH ${LLVM_TOOLS_BINARY_DIR})

set(MLIR_TABLEGEN_EXE mlir-tblgen)

set(MLIR_INCLUDE_DIR ${LLVM_BUILD_MAIN_SRC_DIR}/../mlir/include)
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIR} ";" ${LLVM_BINARY_DIR}/tools/mlir/include PARENT_SCOPE)
set(MLIR_MAIN_SRC_DIR ${LLVM_BUILD_MAIN_SRC_DIR}/../mlir/lib)

function(mlir_tablegen ofn)
    message(STATUS "Running MLIR tablegen" ${MLIR_INCLUDE_DIR})
    tablegen(MLIR ${ARGV} "-I${MLIR_MAIN_SRC_DIR}" "-I=${MLIR_INCLUDE_DIR}")
    message(STATUS "TableGen Output: " ${TABLEGEN_OUTPUT})
    set(TABLEGEN_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
            PARENT_SCOPE)
endfunction()

set(LLVM_TARGET_DEFINITIONS SPN_Ops.td)
mlir_tablegen(SPNOps.h.inc -gen-op-decls)
message(STATUS "TableGen: " ${TABLEGEN_OUTPUT})
set(TABLEGEN_HEADER ${TABLEGEN_OUTPUT} PARENT_SCOPE)
mlir_tablegen(SPNOps.cpp.inc -gen-op-defs)
set(TABLEGEN_IMPL ${TABLEGEN_OUTPUT} PARENT_SCOPE)

add_public_tablegen_target(SPNOpsIncGen)