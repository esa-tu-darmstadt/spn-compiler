add_mlir_conversion_library(MLIRSPNtoStandardTransforms
        SPNtoStandardPatterns.cpp
        SPNtoStandardConversionPass.cpp
        Vectorization/BatchVectorization/BatchVectorizationPatterns.cpp
        Vectorization/BatchVectorization/QueryBatchVectorization.cpp
        
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Conversion

        LINK_LIBS PUBLIC
        MLIRSPN
        MLIRIR
        MLIRMath
        MLIRPass
        MLIRStandard
        MLIRSupport
        MLIRTransforms

        DEPENDS
        cpu_targetinfo
        )

        target_link_libraries(MLIRSPNtoStandardTransforms PRIVATE cpu_targetinfo)
