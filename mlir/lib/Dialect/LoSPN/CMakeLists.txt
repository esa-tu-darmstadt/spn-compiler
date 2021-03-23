add_mlir_dialect_library(MLIRLoSPN
        LoSPNDialect.cpp
        LoSPNOps.cpp
        LoSPNAttributes.cpp
        Passes/LoSPNBufferize.cpp
        Bufferize/LoSPNBufferizationPatterns.cpp
        Analysis/SPNGraphStatistics.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Dialect

        DEPENDS
        MLIRLoSPNOpsIncGen
        MLIRLoSPNAttributesIncGen
        MLIRLoSPNInterfacesIncGen
        MLIRLoSPNPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        )