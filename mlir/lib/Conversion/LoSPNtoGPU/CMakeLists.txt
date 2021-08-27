# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

message(STATUS "CUDA include dir: ${CUDA_INCLUDE_DIRS}")

add_subdirectory(Target)

add_mlir_conversion_library(MLIRLoSPNtoGPUTransforms
        GPUNodePatterns.cpp
        GPUStructurePatterns.cpp
        LoSPNtoGPUConversionPasses.cpp
        LoSPNGPUSharedMemoryInsertion.cpp
        Passes/GPUBufferDeallocation.cpp
        Passes/GPUCopyElimination.cpp

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
        MLIRGPU

        DEPENDS
        MLIRLoSPNtoGPUPassesIncGen
        cuda_targetinfo
        )

target_link_libraries(MLIRLoSPNtoGPUTransforms PRIVATE cuda_targetinfo)