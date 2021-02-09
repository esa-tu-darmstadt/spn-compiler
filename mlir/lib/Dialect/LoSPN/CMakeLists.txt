add_mlir_dialect_library(MLIRLoSPN
        LoSPNDialect.cpp
        LoSPNOps.cpp
        LoSPNAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Dialect

        DEPENDS
        MLIRLoSPNOpsIncGen
        MLIRLoSPNAttributesIncGen
        MLIRLoSPNInterfacesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransformUtils
        )