add_mlir_dialect_library(MLIRSPN
        SPNDialect.cpp
        SPNOps.cpp
        SPNAttributes.cpp
        canonicalization/CanonicalizationPatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir-dialects/include

        DEPENDS
        MLIRSPNOpsIncGen
        MLIRSPNAttributesIncGen
        MLIRSPNInterfacesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
