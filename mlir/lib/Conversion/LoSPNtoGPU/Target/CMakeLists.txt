add_mlir_library(cuda_targetinfo
        SHARED
        CUDATargetInformation.cpp

        LINK_LIBS PUBLIC
        MLIRIR

        EXCLUDE_FROM_LIBMLIR
        )

target_include_directories(cuda_targetinfo PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(cuda_targetinfo PRIVATE ${CUDA_RUNTIME_LIBRARY})