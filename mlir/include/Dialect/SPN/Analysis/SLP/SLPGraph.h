//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPGRAPH_H
#define SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPGRAPH_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
  namespace spn {
    namespace slp {

      ///
      /// Graph class storing Use-Def chains of an SPN.
      class SLPGraph {

      public:

        /// Constructor, initialize analysis.
        /// \param root Root node of a (sub-)graph or query operation.
        explicit SLPGraph(Operation* root);

        ///
        /// \return All seed operations.
        const SmallPtrSet<Operation, 16>& getSeeds() const;

      private:

        void analyzeGraph(Operation* root);
        void traverseSubgraph(Operation* root);

        SmallPtrSet<Operation, 16> seeds;

      };
    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPGRAPH_H
