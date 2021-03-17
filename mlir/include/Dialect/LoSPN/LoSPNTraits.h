//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNTRAITS_H
#define SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNTRAITS_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
  namespace OpTrait {
    namespace spn {
      namespace low {

        template<typename ConcreteType>
        class VectorizableOp : public TraitBase<ConcreteType, VectorizableOp> {

        public:

          static StringRef getVectorWidthAttrName() { return "vector_width"; }

          bool isVectorized() {
            if (!this->getOperation()->hasAttr(getVectorWidthAttrName())) {
              return false;
            }
            auto VF = this->getOperation()->template getAttrOfType<IntegerAttr>(getVectorWidthAttrName()).getInt();
            return VF != 0;
          }

          unsigned vectorFactor() {
            if (!this->getOperation()->hasAttr(getVectorWidthAttrName())) {
              return 0;
            }
            return this->getOperation()->template getAttrOfType<IntegerAttr>(getVectorWidthAttrName()).getInt();
          }

          void vectorFactor(unsigned _vectorFactor) {
            if (_vectorFactor == 0) {
              if (this->getOperation()->hasAttr(getVectorWidthAttrName())) {
                this->getOperation()->removeAttr(getVectorWidthAttrName());
              }
              return;
            }
            auto VFAttr = IntegerAttr::get(IntegerType::get(this->getOperation()->getContext(), 32),
                                           _vectorFactor);
            this->getOperation()->setAttr(getVectorWidthAttrName(), VFAttr);
          }

        };

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNTRAITS_H
