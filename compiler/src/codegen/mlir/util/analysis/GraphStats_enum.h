//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_GRAPHSTATS_ENUM_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_GRAPHSTATS_ENUM_H

#include <initializer_list>

enum class NODETYPE { SUM, PRODUCT, HISTOGRAM };

// Allow to conveniently iterate over enum classes.
constexpr std::initializer_list<NODETYPE> LISTOF_NODETYPE = { NODETYPE::SUM, NODETYPE::PRODUCT, NODETYPE::HISTOGRAM };
constexpr std::initializer_list<NODETYPE> LISTOF_NODETYPE_INNER = { NODETYPE::SUM, NODETYPE::PRODUCT };
constexpr std::initializer_list<NODETYPE> LISTOF_NODETYPE_LEAF = { NODETYPE::HISTOGRAM };

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_GRAPHSTATS_ENUM_H
