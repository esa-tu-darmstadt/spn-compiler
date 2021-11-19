//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_INCLUDE_SPN_SPNOPTRAITS_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_INCLUDE_SPN_SPNOPTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
  namespace OpTrait {
    namespace spn {

      template<typename ConcreteType>
      class Vectorizable : public TraitBase<ConcreteType, Vectorizable> {};
    }
  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_INCLUDE_SPN_SPNOPTRAITS_H
