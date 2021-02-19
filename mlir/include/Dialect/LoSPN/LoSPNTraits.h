//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNTRAITS_H
#define SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
  namespace OpTrait {
    namespace spn {
      namespace low {

        template<typename ConcreteType>
        class VectorizableOp : public TraitBase<ConcreteType, VectorizableOp> {

        public:

          bool isVectorized() { return VF != 0; }

          unsigned vectorFactor() { return VF; }

          void vectorFactor(unsigned _vectorFactor) {
            VF = _vectorFactor;
          }

        private:

          unsigned VF = 0;

        };

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNTRAITS_H
