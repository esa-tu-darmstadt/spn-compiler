# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

add_mlir_conversion_library(MLIRHiSPNtoLoSPNTransforms
        NodePatterns.cpp
        QueryPatterns.cpp
        HiSPNtoLoSPNConversionPasses.cpp
        Analysis/ArithmeticPrecisionAnalysis.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Conversion

        LINK_LIBS PUBLIC
        MLIRHiSPN
        MLIRLoSPN
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRTransforms
        )