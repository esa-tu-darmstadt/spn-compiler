add_mlir_conversion_library(MLIRLoSPNtoGPUTransforms
        GPUNodePatterns.cpp
        GPUStructurePatterns.cpp
        LoSPNtoGPUConversionPasses.cpp

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
        )