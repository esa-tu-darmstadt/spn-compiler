add_mlir_dialect_library(MLIRSPN
        SPNDialect.cpp
        SPNOps.cpp
        SPNAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ../../include

        DEPENDS
        MLIRSPNOpsIncGen
        MLIRSPNAttributesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
