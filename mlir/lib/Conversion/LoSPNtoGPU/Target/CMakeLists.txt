add_mlir_library(cuda_targetinfo
        
        CUDATargetInformation.cpp

        LINK_LIBS PUBLIC
        MLIRIR

        EXCLUDE_FROM_LIBMLIR
        )

target_include_directories(cuda_targetinfo PRIVATE ${CUDA_INCLUDE_DIRS})

# For some odd reason, the target_include_directories above is not 
# sufficient when building a shared library, therefore we need
# the following additional statement.
include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(cuda_targetinfo PRIVATE ${CUDA_RUNTIME_LIBRARY})