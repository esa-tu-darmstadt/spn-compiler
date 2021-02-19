add_mlir_conversion_library(MLIRLoSPNtoCPUTransforms
        StructurePatterns.cpp
        NodePatterns.cpp
        Vectorization/VectorizeStructurePatterns.cpp
        Vectorization/VectorizeNodePatterns.cpp
        Vectorization/TargetInformation.cpp
        LoSPNtoCPUConversionPasses.cpp

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
        )