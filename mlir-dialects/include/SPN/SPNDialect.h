//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_INCLUDE_SPN_SPNDIALECT_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_INCLUDE_SPN_SPNDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
  namespace spn {

    class ProbabilityType : public Type::TypeBase<ProbabilityType, Type, TypeStorage> {
    public:
      using Base::Base;
    };

  }
}

#include "SPN/SPNOpsDialect.h.inc"

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_INCLUDE_SPN_SPNDIALECT_H
