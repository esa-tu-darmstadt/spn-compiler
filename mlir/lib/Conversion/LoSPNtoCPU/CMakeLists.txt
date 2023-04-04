# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

add_subdirectory(Target)

add_mlir_conversion_library(MLIRLoSPNtoCPUTransforms
        StructurePatterns.cpp
        NodePatterns.cpp
        Vectorization/VectorizeStructurePatterns.cpp
        Vectorization/VectorizeNodePatterns.cpp
        Vectorization/ReplaceGatherWithShuffle.cpp
        Vectorization/Util.cpp
        Vectorization/SLP/CostModel.cpp
        Vectorization/SLP/GraphConversion.cpp
        Vectorization/SLP/PatternVisitors.cpp
        Vectorization/SLP/ScoreModel.cpp
        Vectorization/SLP/Seeding.cpp
        Vectorization/SLP/SLPGraph.cpp
        Vectorization/SLP/SLPGraphBuilder.cpp
        Vectorization/SLP/SLPPatternMatch.cpp
        Vectorization/SLP/SLPVectorizationPatterns.cpp
        Vectorization/SLP/Util.cpp
        LoSPNtoCPUConversionPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Conversion

        LINK_LIBS PUBLIC
        MLIRLoSPN
        MLIRIR
        MLIRPass
        #MLIRStandard
        MLIRSupport
        MLIRTransforms
        MLIRMathDialect

        DEPENDS
        cpu_targetinfo
        LLVMTargetParser
        )

target_link_libraries(MLIRLoSPNtoCPUTransforms PRIVATE cpu_targetinfo LLVMTargetParser)

if (${SPNC_SLP_DEBUG})
    # Usually, target_compile_definitions should be preferred over the following command,
    # but LLVM libraries seem to not accept that.
    add_compile_definitions(SLP_DEBUG=true)
endif (${SPNC_SLP_DEBUG})