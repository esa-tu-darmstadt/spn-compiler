//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_DIALECT_HISPN_HISPNDIALECT_H
#define SPNC_MLIR_INCLUDE_DIALECT_HISPN_HISPNDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
  namespace spn {
    namespace high {

      ///
      /// Abstract type representing probability values computed inside an SPN.
      /// The actual datatype to use for computation will be determined through
      /// floating point error analysis.
      class ProbabilityType : public Type::TypeBase<ProbabilityType, Type, TypeStorage> {
      public:
        using Base::Base;
      };

    }
  }
}

#include "HiSPN/HiSPNOpsDialect.h.inc"

#endif //SPNC_MLIR_INCLUDE_DIALECT_HISPN_HISPNDIALECT_H
