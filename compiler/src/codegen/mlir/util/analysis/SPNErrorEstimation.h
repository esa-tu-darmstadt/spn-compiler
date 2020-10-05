//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_SPNERRORESTIMATION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_SPNERRORESTIMATION_H

#include <codegen/mlir/dialects/spn/SPNDialect.h>

namespace mlir {
  namespace spn {

    enum class ERRORMODEL { EM_FIXED, EM_FLOATING };

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_SPNERRORESTIMATION_H