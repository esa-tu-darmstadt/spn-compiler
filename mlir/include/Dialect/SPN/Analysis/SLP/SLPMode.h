//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPMODE_H
#define SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPMODE_H

#include "SPN/SPNOps.h"

namespace mlir {
  namespace spn {
    namespace slp {

      enum MODE {
        // look for a constant
        CONST,
        // look for a consecutive load to that in the previous lane
        // TODO: do we have load operations? or something that can be treated equivalently?
        LOAD,
        // look for an operation of the same opcode
        OPCODE,
        // look for the exact same operation
        // TODO: determine whether SPLAT mode is actually relevant (are the same operands used in more than one lane?)
        SPLAT,
        // vectorization has failed, give higher priority to others
        FAILED
      };

      static MODE modeFromOperation(Operation const* operation) {
        if (dyn_cast<ConstantOp>(operation)) {
          return CONST;
        }
        // We don't have LOADs. Therefore just return OPCODE.
        return OPCODE;
      }

    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPMODE_H
