# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

add_mlir_dialect_library(MLIRLoSPN
        LoSPNDialect.cpp
        LoSPNOps.cpp
        LoSPNAttributes.cpp
        Passes/LoSPNBufferize.cpp
        Passes/LoSPNCopyRemoval.cpp
        Passes/LoSPNGraphStatsCollection.cpp
        Passes/LoSPNTaskPartitioner.cpp
        Passes/ReplaceARMOptimizedRoutines.cpp
        Bufferize/LoSPNBufferizationPatterns.cpp
        Analysis/SPNGraphStatistics.cpp
        Analysis/SPNNodeLevel.cpp
        Partitioning/GraphPartitioner.cpp
        Partitioning/Heuristic.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Dialect

        DEPENDS
        MLIRLoSPNOpsIncGen
        MLIRLoSPNAttributesIncGen
        MLIRLoSPNInterfacesIncGen
        MLIRLoSPNPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        MLIRMath
        MLIRGPU
        )

include_directories(${PROJECT_SOURCE_DIR}/common/include)