# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

macro(cuda_setup)
    # TODO: The old FindCUDA is deprecated, the new FindCUDAToolkit is only available in version > 3.17.
    # Use this feature here after upgrading the required CMake version.
    find_package(CUDA)
    if (NOT CUDA_FOUND)
        message(FATAL_ERROR "Targeting CUDA GPUs requires a working CUDA install, try setting CUDA_TOOLKIT_ROOT_DIR")
    else ()
        message(STATUS "Using CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} from ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "Using CUDA runtime headers: " ${CUDA_INCLUDE_DIRS})
    endif ()

    # When compiling for CUDA GPUs, the compiler will invoke the PTX compiler and linker
    # through the CUDA runtime library API.
    find_library(CUDA_RUNTIME_LIBRARY cuda)
    if (NOT CUDA_RUNTIME_LIBRARY)
        message(FATAL_ERROR "CUDA runtime library not found. Set location manually through CUDA_RUNTIME_LIBRARY")
    else ()
        message(STATUS "Using CUDA runtime library: " ${CUDA_RUNTIME_LIBRARY})
    endif ()

    # When compiling for CUDA GPUs, the generated Kernel must be linked with the
    # MLIR CUDA runtime wrappers for data-transfer, kernel-launch etc.
    # When building static libraries, we want to create a complete Python package, and
    # include the MLIR CUDA wrapper library, which the compiled kernels for the GPU target need
    # in the Python wheel.
    find_library(MLIR_CUDA_WRAPPERS cuda-runtime-wrappers HINTS ${LLVM_BUILD_LIBRARY_DIR})
    if (NOT MLIR_CUDA_WRAPPERS)
        message(FATAL_ERROR "MLIR CUDA wrappers not found.")
    else ()
        # Resolve symbolic link if necessary.
        if(IS_SYMLINK ${MLIR_CUDA_WRAPPERS})
            file(READ_SYMLINK ${MLIR_CUDA_WRAPPERS} MLIR_CUDA_SYM)
            if(NOT IS_ABSOLUTE "${MLIR_CUDA_SYM}")
                get_filename_component(dir "${MLIR_CUDA_WRAPPERS}" DIRECTORY)
                set(MLIR_CUDA_WRAPPERS "${dir}/${MLIR_CUDA_SYM}")
            else()
                set(MLIR_CUDA_WRAPPERS "${MLIR_CUDA_SYM}")
            endif()
        endif()
        message(STATUS "Using MLIR CUDA wrappers: ${MLIR_CUDA_WRAPPERS}")
    endif()

endmacro(cuda_setup)