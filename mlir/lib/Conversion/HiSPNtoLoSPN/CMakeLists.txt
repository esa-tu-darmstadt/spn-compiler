add_mlir_conversion_library(MLIRHiSPNtoLoSPNTransforms
        NodePatterns.cpp
        QueryPatterns.cpp
        HiSPNtoLoSPNConversionPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Conversion

        LINK_LIBS PUBLIC
        MLIRHiSPN
        MLIRLoSPN
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRTransforms
        )