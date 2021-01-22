//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPMODE_H
#define SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPMODE_H

namespace mlir {
  namespace spn {
    namespace slp {

      enum MODE {
        // look for a constant
        CONST,
        // look for a consecutive load to that in the previous lane
        LOAD,
        // look for an operation of the same opcode
        OPCODE,
        // look for the exact same operation
        SPLAT,
        // vectorization has failed, give higher priority to others
        FAILED
      };

    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPMODE_H
