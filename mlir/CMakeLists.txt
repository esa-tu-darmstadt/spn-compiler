# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

find_package(CIRCT REQUIRED CONFIG)
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")

if (POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
    set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif ()

if (POLICY CMP0075)
    cmake_policy(SET CMP0075 NEW)
endif ()

if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/Dialect)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/Conversion)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/Conversion)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

link_directories(${CIRCT_BUILD_LIBRARY_DIR})
add_definitions(${CIRCT_DEFINITIONS})

set(SPN_DIALECT_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Dialect
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Conversion
        ${CMAKE_CURRENT_BINARY_DIR}/include/Dialect
        ${CMAKE_CURRENT_BINARY_DIR}/include/Conversion
        PARENT_SCOPE)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

doxygen_doc(TARGET_NAME mlir-doc
        SRC_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}/include/Dialect)