add_library(spnc-cuda-wrappers SHARED

        CUDARuntimeWrappers.cpp
        )


target_include_directories(spnc-cuda-wrappers PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(spnc-cuda-wrappers PRIVATE ${CUDA_RUNTIME_LIBRARY})

option(CUDA_UNIFIED_MEMORY
        "Enable/Disable support for unified memory usage for CUDA GPUs."
        OFF
        )

if (${CUDA_UNIFIED_MEMORY})
    target_compile_definitions(spnc-cuda-wrappers PRIVATE SPNC_CUDA_UNIFIED_MEMORY=1)
else (${CUDA_UNIFIED_MEMORY})
    target_compile_definitions(spnc-cuda-wrappers PRIVATE SPNC_CUDA_UNIFIED_MEMORY=0)
endif (${CUDA_UNIFIED_MEMORY})