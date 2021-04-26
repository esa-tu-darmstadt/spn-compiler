add_mlir_dialect_library(MLIRLoSPN
        LoSPNDialect.cpp
        LoSPNOps.cpp
        LoSPNAttributes.cpp
        Passes/LoSPNBufferize.cpp
        Passes/LoSPNGraphStatsCollection.cpp
        Bufferize/LoSPNBufferizationPatterns.cpp
        Analysis/SPNGraphStatistics.cpp
        Analysis/SPNNodeLevel.cpp

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

include_directories(${PROJECT_SOURCE_DIR}/common/include)