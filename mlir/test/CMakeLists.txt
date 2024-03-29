# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

add_custom_target(check-spnc-mlir)

# Note: Here [CACHE INTERNAL ""] is used to make the set variable globally available.
set(SPNC_MLIR_TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "")
set(SPNC_MLIR_TEST_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "")
set(SPNC_OPT_TOOL_BINARY_PATH "${CMAKE_BINARY_DIR}/bin/spnc-opt" CACHE INTERNAL "")

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py

        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py

        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# (Re-)Set llvm-lit arguments
set(CHECK_SPNC_LIT_ARGS "")

# Add verbose flag if requested
if (${CHECK_SPNC_VERBOSE})
    message(STATUS "Running llvm-lit in verbose mode")
    set(CHECK_SPNC_LIT_ARGS "${CHECK_SPNC_LIT_ARGS};-v")
endif ()

# Mark the directory containing the GPU/CUDA tests as unsupported if no CUDA GPU support was requested.
if (NOT ${CUDA_GPU_SUPPORT})
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/transform/gpu/cuda/lit.local.cfg config.unsupported=True)
else ()
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/transform/gpu/cuda/lit.local.cfg config.unsupported=False)
endif ()


# Run regression tests
add_lit_testsuite(check-spnc-mlir-tests "Running spnc-mlir regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        PARAMS spnc_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
        DEPENDS spnc-opt driver
        ARGS ${CHECK_SPNC_LIT_ARGS})
set_target_properties(check-spnc-mlir-tests PROPERTIES FOLDER "Tests")

add_dependencies(check-spnc-mlir check-spnc-mlir-tests)
