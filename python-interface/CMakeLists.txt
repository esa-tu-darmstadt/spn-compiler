# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

find_package(pybind11 REQUIRED)

pybind11_add_module(spncpy MODULE src/python-interface.cpp)

target_link_libraries(spncpy PUBLIC spnc-rt spnc-common spnc)

target_include_directories(spncpy
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)


# Create the setup.py and supplement some information from the CMake configure and generate stage.
# This is a multi-stage process, because we first need to use configure_file to supplement 
# configuration variables (used for the MLIR CUDA runtime wrapper library) and then need 
# to substiture the generator expressions used to add target information to the setup.py
if (${CUDA_GPU_SUPPORT})
    set(INCLUDE_MLIR_CUDA_WRAPPERS "True")
    # Copy MLIR CUDA wrappers to spnc package directory.
    add_custom_command(TARGET spncpy POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${MLIR_CUDA_WRAPPERS} ${CMAKE_CURRENT_SOURCE_DIR}/spnc)
    get_filename_component(MLIR_CUDA_WRAPPER_NAME ${MLIR_CUDA_WRAPPERS} NAME)
else ()
    set(INCLUDE_MLIR_CUDA_WRAPPERS "False")
    set(MLIR_CUDA_WRAPPER_NAME "")
endif ()

# Copy the spnc Python interface module/library to the spnc package directory.
add_custom_command(TARGET spncpy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:spncpy> ${CMAKE_CURRENT_SOURCE_DIR}/spnc)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py.in)

file(GENERATE OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/setup.py INPUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py.in)