# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

add_mlir_dialect(HiSPNOps hi_spn)
add_mlir_doc(HiSPNDialect -gen-dialect-doc HiSPNDialect HiSPN/)
add_mlir_doc(HiSPNOps -gen-op-doc HiSPNOps HiSPN/)

set(LLVM_TARGET_DEFINITIONS HiSPNOps.td)
mlir_tablegen(HiSPNAttributes.h.inc -gen-struct-attr-decls)
mlir_tablegen(HiSPNAttributes.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRHiSPNAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS HiSPNOps.td)
mlir_tablegen(HiSPNEnums.h.inc -gen-enum-decls)
mlir_tablegen(HiSPNEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHiSPNEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS HiSPNInterfaces.td)
mlir_tablegen(HiSPNInterfaces.h.inc -gen-op-interface-decls -name HiSPN)
mlir_tablegen(HiSPNInterfaces.cpp.inc -gen-op-interface-defs -name HiSPN)
add_public_tablegen_target(MLIRHiSPNInterfacesIncGen)