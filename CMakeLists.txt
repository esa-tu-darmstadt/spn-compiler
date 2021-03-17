cmake_minimum_required(VERSION 3.13.4 FATAL_ERROR)

project(spnc VERSION 0.0.4 LANGUAGES CXX C)

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

add_subdirectory(common)
# Ensure processing of ./execute before ./compiler
# Reason: tests in ./compiler depend on CMake variables set in ./execute
add_subdirectory(execute)
add_subdirectory(mlir)
add_subdirectory(compiler)
add_subdirectory(compiler-rt)
add_subdirectory(runtime)
add_subdirectory(python-interface)

doxygen_toplevel()
