//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_INCLUDE_UTIL_LOGGING_H
#define SPNC_COMPILER_INCLUDE_UTIL_LOGGING_H

// Required to enable setting verbose log-levels.
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <spdlog/spdlog.h>

#define SPNC_FATAL_ERROR(...)                                                  \
  SPDLOG_ERROR(__VA_ARGS__);                                                   \
  throw std::runtime_error("COMPILATION ERROR");

#define SPN_RT_FATAL_ERROR(...)                                                \
  SPDLOG_ERROR(__VA_ARGS__);                                                   \
  throw std::runtime_error("RUNTIME ERROR");

#endif // SPNC_COMPILER_INCLUDE_UTIL_LOGGING_H
