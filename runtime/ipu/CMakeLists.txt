# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

add_library(spnc-ipu-rt IPURuntime.cpp)

target_link_libraries(spnc-ipu-rt PRIVATE elf spnc-common spdlog::spdlog poplar)

target_compile_options(spnc-rt PRIVATE ${SPNC_CXX_WARNING_LEVEL})