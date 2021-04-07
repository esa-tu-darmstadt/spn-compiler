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

    # When building static libraries, we want to create a complete Python package, including 
    # all non-standard libraries. This includes the MLIR CUDA wrapper library, which the 
    # compiled kernels for the GPU target need.
    if (NOT BUILD_SHARED_LIBS)
        find_library(MLIR_CUDA_WRAPPERS cuda-runtime-wrappers HINTS "${LLVM_BINARY_DIR}/lib")
        if (NOT MLIR_CUDA_WRAPPERS)
            message(FATAL_ERROR "MLIR CUDA wrappers not found.")
        else ()
            message(STATUS "Using MLIR CUDA wrappers: ${MLIR_CUDA_WRAPPERS}")
        endif()
    endif()
endmacro(cuda_setup)