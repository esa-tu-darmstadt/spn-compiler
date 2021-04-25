//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPSEEDING_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPSEEDING_H

#include "mlir/IR/Operation.h"
#include "SLPNode.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        enum SearchMode {
          ///
          DefBeforeUse,
          ///
          UseBeforeDef,
          /// TODO?
          Chain
        };

        class SeedAnalysis {

        public:

          explicit SeedAnalysis(Operation* rootOp);

          ArrayRef<Value> getSeed(unsigned width, SearchMode const& mode) const;

        private:
          Operation* rootOp;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPSEEDING_H
