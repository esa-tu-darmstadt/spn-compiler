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
#include "SLPNode.h"

#include <vector>
#include <set>

namespace mlir {
  namespace spn {
    namespace slp {

      ///
      /// Graph class storing Use-Def chains of an SPN.
      class SLPTree {

      public:

        /// Constructor, initialize analysis.
        /// \param root Root node of a (sub-)graph or query operation.
        explicit SLPTree(Operation* op);

      private:

        void buildGraph(std::vector<Operation*> const& values);

        bool vectorizable(std::vector<Operation*> const& values) const;
        bool commutative(std::vector<Operation*> const& values) const;
        bool attachableOperands(Operation* operation) const;

        SLPNode graph;

      };
    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPGRAPH_H
