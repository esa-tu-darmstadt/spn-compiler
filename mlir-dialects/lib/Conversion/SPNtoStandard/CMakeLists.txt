add_mlir_conversion_library(MLIRSPNtoStandardTransforms
        SPNtoStandardPatterns.cpp
        SPNtoStandardConversionPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir-dialects/include/Conversion

        LINK_LIBS PUBLIC
        MLIRSPN
        MLIRIR
        MLIRPass
        MLIRStandard
        MLIRSupport
        MLIRTransforms
        )