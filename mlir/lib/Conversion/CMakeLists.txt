# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

add_subdirectory(HiSPNtoLoSPN)
add_subdirectory(LoSPNtoCPU)
if (${CUDA_GPU_SUPPORT})
    add_subdirectory(LoSPNtoGPU)
endif (${CUDA_GPU_SUPPORT})
add_subdirectory(LoSPNtoFPGA)
