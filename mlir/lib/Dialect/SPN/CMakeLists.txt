add_mlir_dialect_library(MLIRSPN
        SPNDialect.cpp
        SPNOps.cpp
        SPNAttributes.cpp
        SPNEnums.cpp
        Analysis/SPNNodeLevel.cpp
        Analysis/SPNGraphStatistics.cpp
        Analysis/SLP/SLPMode.cpp
        Analysis/SLP/SLPNode.cpp
        Analysis/SLP/SLPTree.cpp
        Analysis/SLP/SLPSeeding.cpp
        Canonicalization/CanonicalizationPatterns.cpp
        Simplification/SimplificationPatterns.cpp
        TypeAnalysis/TypePinningPatterns.cpp
        Passes/SPNOpSimplifier.cpp
        Passes/SPNTypePinning.cpp
        Passes/SPNVectorization.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Dialect

        DEPENDS
        MLIRSPNOpsIncGen
        MLIRSPNAttributesIncGen
        MLIRSPNInterfacesIncGen
        MLIRSPNEnumsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransformUtils
        )
