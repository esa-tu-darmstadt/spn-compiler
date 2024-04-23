# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

add_mlir_dialect(LoSPNOps lo_spn)
add_mlir_doc(LoSPNDialect LoSPNDialect LoSPN/ -gen-dialect-doc)
add_mlir_doc(LoSPNOps LoSPNOps LoSPN/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS LoSPNAttributes.td)
mlir_tablegen(LoSPNAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(LoSPNAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRLoSPNAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS LoSPNInterfaces.td)
mlir_tablegen(LoSPNInterfaces.h.inc -gen-op-interface-decls -name LoSPN)
mlir_tablegen(LoSPNInterfaces.cpp.inc -gen-op-interface-defs -name LoSPN)
add_public_tablegen_target(MLIRLoSPNInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS LoSPNPasses.td)
mlir_tablegen(LoSPNPasses.h.inc -gen-pass-decls -name LoSPN)
add_public_tablegen_target(MLIRLoSPNPassesIncGen)