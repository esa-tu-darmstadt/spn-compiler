# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

cmake_minimum_required(VERSION 3.13.4 FATAL_ERROR)

project(spnc VERSION 0.1 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(spdlog REQUIRED)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include(Doxygen.cmake)
doxygen_setup()

option(CUDA_GPU_SUPPORT
        "Enable/disable support for CUDA GPUs. Requires a working installation of the CUDA driver and toolkit"
        OFF
        )

if (${CUDA_GPU_SUPPORT})
    include(CudaToolkit.cmake)
    cuda_setup()
endif (${CUDA_GPU_SUPPORT})

set(SPNC_CXX_WARNING_LEVEL "-Wall")

#
# clang-tidy setup
#
string(CONCAT SPNC_CLANG_TIDY_CHECKS
        "bugprone-*,"
        "modernize-*,-modernize-use-trailing-return-type,-modernize-avoid-c-arrays,"
        "performance-*,"
        "clang-analyzer-*,"
        "readibility-*,"
        "google-default-arguments,google-explicit-constructor,google-runtime-operator,"
        "misc-misplaced-const,misc-new-delete-overloads,misc-non-copyable-objects,"
        "misc-throw-by-value-catch-by-reference,misc-unconventional-assign-operator,"
        "misc-uniqueptr-reset-release,"
        "cppcoreguidelines-init-variables,"
        "cppcoreguidelines-interfaces-global-init,"
        "cppcoreguidelines-narrowing-conversions,"
        "cppcoreguidelines-pro-type-member-init,"
        "cppcoreguidelines-pro-type-static-cast-downcast,"
        "cppcoreguidelines-slicing")

option(SPNC_ENABLE_CLANG_TIDY
        "Run clang-tidy as part of the build process. May slow down your build. Requires clang-tidy to be on the PATH"
        OFF
        )

if(${SPNC_ENABLE_CLANG_TIDY})
    if(${CMAKE_EXPORT_COMPILE_COMMANDS})
        find_program(CLANG_TIDY_EXE "clang-tidy")
        if (CLANG_TIDY_EXE-NOTFOUND)
            message(WARNING "clang-tidy not found")
        else ()
            message(STATUS "Using clang tidy from ${CLANG_TIDY_EXE}")
            set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=${SPNC_CLANG_TIDY_CHECKS}")
        endif ()
    else ()
        message(WARNING "CMAKE_EXPORT_COMPILE_COMMANDS must be enabled in order to run clang-tidy")
    endif ()
endif (${SPNC_ENABLE_CLANG_TIDY})

option(SPNC_RUN_CLANG_TIDY_LOCATION
        "Location of the run-clang-tidy script used as a hint to find the script"
        "/usr/bin")

if (${CMAKE_EXPORT_COMPILE_COMMANDS})
    find_program(RUN_CLANG_TIDY_SCRIPT
            NAMES run-clang-tidy run-clang-tidy.py
            HINTS ${SPNC_RUN_CLANG_TIDY_LOCATION})
    if (NOT RUN_CLANG_TIDY_SCRIPT)
        message(WARNING "run-clang-tidy not found, not generating clang-tidy wrapper script")
    else ()
        message(STATUS "Using run-clang-tidy from ${RUN_CLANG_TIDY_SCRIPT}")
        # Configure script with run-clang-tidy executable and checks.
        configure_file(${PROJECT_SOURCE_DIR}/wrap-clang-tidy.sh.in
                ${CMAKE_CURRENT_BINARY_DIR}/clang-tidy/wrap-clang-tidy.sh)
        # Make script executable
        file(COPY ${CMAKE_CURRENT_BINARY_DIR}/clang-tidy/wrap-clang-tidy.sh
                DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    endif()
endif()

add_subdirectory(common)
add_subdirectory(mlir)
add_subdirectory(compiler)
add_subdirectory(runtime)
add_subdirectory(execute)
add_subdirectory(python-interface)

doxygen_toplevel()
