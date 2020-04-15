add_custom_target(check-spnc-mlir)

# Run regression tests
add_lit_testsuite(check-spnc-mlir-tests "Running spnc-mlir regression tests"
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS spnc-opt)
set_target_properties(check-spnc-mlir-tests PROPERTIES FOLDER "Tests")

add_dependencies(check-spnc-mlir check-spnc-mlir-tests)
