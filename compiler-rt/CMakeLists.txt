#
# This file is part of the SPNC project.
# Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
#
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_definitions(${LLVM_DEFINITIONS})

include(LLVMBitcodeLib.cmake)

add_bitcode_library(LIBNAME compiler-rt SOURCES src/trace.cpp)