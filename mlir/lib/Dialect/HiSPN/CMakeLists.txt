add_mlir_dialect_library(MLIRHiSPN
        HiSPNDialect.cpp
        HiSPNOps.cpp
        HiSPNAttributes.cpp
        HiSPNEnums.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Dialect

        DEPENDS
        MLIRHiSPNOpsIncGen
        MLIRHiSPNAttributesIncGen
        MLIRHiSPNInterfacesIncGen
        MLIRHiSPNEnumsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransformUtils
        )