message(STATUS "CUDA include dir: ${CUDA_INCLUDE_DIRS}")

add_subdirectory(Target)

add_mlir_conversion_library(MLIRLoSPNtoGPUTransforms
        GPUNodePatterns.cpp
        GPUStructurePatterns.cpp
        LoSPNtoGPUConversionPasses.cpp
        LoSPNGPUSharedMemoryInsertion.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Conversion

        LINK_LIBS PUBLIC
        MLIRLoSPN
        MLIRIR
        MLIRPass
        MLIRStandard
        MLIRSupport
        MLIRTransforms
        MLIRMath
        MLIRLinalg
        MLIRGPU

        DEPENDS
        cuda_targetinfo
        )

target_link_libraries(MLIRLoSPNtoGPUTransforms PRIVATE cuda_targetinfo)