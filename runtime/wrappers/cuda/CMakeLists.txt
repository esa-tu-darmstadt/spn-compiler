add_library(spnc-cuda-wrappers SHARED

        CUDARuntimeWrappers.cpp
        )


target_include_directories(spnc-cuda-wrappers PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(spnc-cuda-wrappers PRIVATE ${CUDA_RUNTIME_LIBRARY})