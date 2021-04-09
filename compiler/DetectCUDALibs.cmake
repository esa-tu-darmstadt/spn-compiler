macro(detect_cuda_libs)
    # When compiling for CUDA GPUs, the LLVM IR generated for GPU kernels must be linked
    # with libdevice, a Nvidia-provided bitcode library with optimized device functions (e.g. log, exp).
    if (NOT SPNC_LIBDEVICE)
        # The libdevice was not explicitly specified by the user.
        message(STATUS "Searching " ${CUDA_TOOLKIT_ROOT_DIR} " for libdevice")
        find_file(SPNC_LIBDEVICE libdevice.10.bc HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES "nvvm/libdevice")
        if (NOT SPNC_LIBDEVICE)
            # We did not find it with its modern name, try to find it
            # using a regex matching the names used in older versions of CUDA
            file(GLOB_RECURSE LIBDEVICE_CANDIDATES CONFIGURE_DEPENDS ${CUDA_TOOLKIT_ROOT_DIR}/libdevice*.bc)
            if (LIBDEVICE_CANDIDATES)
                list(GET LIBDEVICE_CANDIDATES 0 SPNC_LIBDEVICE)
            else ()
                message(FATAL_ERROR "Could not find libdevice, specify by defining SPNC_LIBDEVICE")
            endif ()
        endif ()
    endif ()
    message(STATUS "Using CUDA libdevice from: " ${SPNC_LIBDEVICE})
endmacro(detect_cuda_libs)