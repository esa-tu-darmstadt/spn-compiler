# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

set(LLVM_TARGET_DEFINITIONS LoSPNtoGPUPasses.td)
mlir_tablegen(LoSPNtoGPUPasses.h.inc -gen-pass-decls -name LoSPNtoGPU)
add_public_tablegen_target(MLIRLoSPNtoGPUPassesIncGen)