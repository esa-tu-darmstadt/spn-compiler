# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

add_executable(driver src/main.cpp)

target_include_directories(driver PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(driver spnc spnc-rt dl spdlog::spdlog spnc-common)
add_dependencies(driver compiler-rt)
add_compile_definitions(TEST_KERNEL_DIR="${CMAKE_CURRENT_BINARY_DIR}")

set(SPNC_DRIVER_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/driver" CACHE INTERNAL "")