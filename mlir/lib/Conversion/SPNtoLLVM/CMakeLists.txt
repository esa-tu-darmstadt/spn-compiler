add_mlir_conversion_library(MLIRSPNtoLLVMTransforms
        SPNtoLLVMPatterns.cpp
        SPNtoLLVMConversionPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Conversion

        LINK_LIBS PUBLIC
        MLIRSPN
        MLIRIR
        MLIRPass
        MLIRStandard
        MLIRSupport
        MLIRTransforms
        MLIRLLVMIR
        MLIRStandardToLLVM
        )