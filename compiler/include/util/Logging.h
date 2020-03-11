//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_INCLUDE_UTIL_LOGGING_H
#define SPNC_COMPILER_INCLUDE_UTIL_LOGGING_H

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE


#include <spdlog/spdlog.h>

#define SPNC_FATAL_ERROR(...) \
  SPDLOG_ERROR(__VA_ARGS__); \
  throw std::runtime_error("COMPILATION ERROR");

#endif //SPNC_COMPILER_INCLUDE_UTIL_LOGGING_H
