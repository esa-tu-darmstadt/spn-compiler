# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

add_library(cpu_targetinfo

        TargetInformation.cpp
        )

target_include_directories(cpu_targetinfo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(cpu_targetinfo MLIRIR)