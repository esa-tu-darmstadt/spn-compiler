# set(LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
# set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")
# set(LLVM_LIT "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")

# ${CMAKE_CURRENT_BINARY_DIR}
# ${CMAKE_CURRENT_SOURCE_DIR}

# set(LIT "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")
# set(LLVM_LIT "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")
# set(LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
# set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")

# message("CMAKE_CURRENT_BINARY_DIR : ${CMAKE_CURRENT_BINARY_DIR}")
# message("CMAKE_CURRENT_SOURCE_DIR : ${CMAKE_CURRENT_SOURCE_DIR}")
# message("LIT                   : ${LLVM_LIT}")
# message("LLVM_LIT              : ${LLVM_LIT}")
# message("LLVM_BINARY_DIR       : ${LLVM_BINARY_DIR}")
# message("LLVM_TOOLS_BINARY_DIR : ${LLVM_TOOLS_BINARY_DIR}")

add_custom_target(check-spnc-mlir)

# Run regression tests
add_lit_testsuite(check-spnc-mlir-tests "Running spnc-mlir regression tests"
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS spnc-opt)
set_target_properties(check-spnc-mlir-tests PROPERTIES FOLDER "Tests")

add_dependencies(check-spnc-mlir check-spnc-mlir-tests)
