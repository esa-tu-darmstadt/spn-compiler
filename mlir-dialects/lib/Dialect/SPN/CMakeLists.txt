add_mlir_dialect_library(MLIRSPN
        SPNDialect.cpp
        SPNOps.cpp
        SPNAttributes.cpp
        Analysis/SPNNodeLevel.cpp
        Analysis/SPNGraphStatistics.cpp
        Canonicalization/CanonicalizationPatterns.cpp
        Simplification/SimplificationPatterns.cpp
        TypeAnalysis/TypePinningPatterns.cpp
        Passes/SPNOpSimplifier.cpp
        Passes/SPNTypePinning.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir-dialects/include/Dialect

        DEPENDS
        MLIRSPNOpsIncGen
        MLIRSPNAttributesIncGen
        MLIRSPNInterfacesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransformUtils
        )
