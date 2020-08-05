add_custom_target(check-spnc-mlir)

# Note: Here [CACHE INTERNAL ""] is used to make the set variable globally available.
set(SPNC_MLIR_TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "")
set(SPNC_MLIR_TEST_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "")

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

# Run regression tests
add_lit_testsuite(check-spnc-mlir-tests "Running spnc-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS spnc_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  DEPENDS spnc-opt
  ARGS ${CHECK_SPNC_LIT_ARGS})
set_target_properties(check-spnc-mlir-tests PROPERTIES FOLDER "Tests")

add_dependencies(check-spnc-mlir check-spnc-mlir-tests)
