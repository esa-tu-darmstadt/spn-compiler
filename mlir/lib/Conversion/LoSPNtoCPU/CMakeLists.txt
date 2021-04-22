add_mlir_conversion_library(MLIRLoSPNtoCPUTransforms
        StructurePatterns.cpp
        NodePatterns.cpp
        Vectorization/VectorizeStructurePatterns.cpp
        Vectorization/VectorizeNodePatterns.cpp
        Vectorization/TargetInformation.cpp
        Vectorization/ReplaceGatherWithShuffle.cpp
        Vectorization/SLP/SLPMode.cpp
        Vectorization/SLP/SLPNode.cpp
        Vectorization/SLP/SLPSeeding.cpp
        Vectorization/SLP/SLPGraphBuilder.cpp
        Vectorization/SLP/SLPUtil.cpp
        Vectorization/SLP/SLPVectorizationPatterns.cpp
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