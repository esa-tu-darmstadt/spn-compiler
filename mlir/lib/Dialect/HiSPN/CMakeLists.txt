# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

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
        LLVMTargetParser
        )