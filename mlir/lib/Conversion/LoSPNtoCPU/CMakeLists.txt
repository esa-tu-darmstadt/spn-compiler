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
        Vectorization/SLP/CostModel.cpp
        Vectorization/SLP/SLPNode.cpp
        Vectorization/SLP/SLPSeeding.cpp
        Vectorization/SLP/SLPGraphBuilder.cpp
        Vectorization/SLP/SLPUtil.cpp
        Vectorization/SLP/SLPVectorizationPatterns.cpp
        Vectorization/SLP/SLPConversion.cpp
        LoSPNtoCPUConversionPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Conversion

        LINK_LIBS PUBLIC
        MLIRLoSPN
        MLIRIR
        MLIRPass
        MLIRStandard
        MLIRSupport
        MLIRTransforms
        MLIRMath
        MLIRLinalg

        DEPENDS
        cpu_targetinfo
        )

target_link_libraries(MLIRLoSPNtoCPUTransforms PRIVATE cpu_targetinfo)

