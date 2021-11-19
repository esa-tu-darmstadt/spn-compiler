# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRHiSPN
        MLIRLoSPN
        MLIRHiSPNtoLoSPNTransforms
        MLIRLoSPNtoCPUTransforms
        MLIROptLib
        cpu_targetinfo
        )

add_llvm_executable(spnc-opt spnc-opt.cpp)

# Set a compiler definition for the CUDA GPU support for use
# of conditional compilation inside source files
if (${CUDA_GPU_SUPPORT})
    target_compile_definitions(spnc-opt PRIVATE SPNC_CUDA_SUPPORT=1)
    target_link_libraries(spnc-opt PRIVATE MLIRLoSPNtoGPUTransforms)
else (${CUDA_GPU_SUPPORT})
    target_compile_definitions(spnc-opt PRIVATE SPNC_CUDA_SUPPORT=0)
endif (${CUDA_GPU_SUPPORT})

llvm_update_compile_flags(spnc-opt)
target_link_libraries(spnc-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(spnc-opt)