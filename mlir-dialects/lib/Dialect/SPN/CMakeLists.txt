add_mlir_dialect_library(MLIRSPN
        SPNDialect.cpp
        SPNOps.cpp
        SPNAttributes.cpp
        canonicalization/CanonicalizationPatterns.cpp
        simplification/SimplificationPatterns.cpp
        passes/SPNOpSimplifier.cpp

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
