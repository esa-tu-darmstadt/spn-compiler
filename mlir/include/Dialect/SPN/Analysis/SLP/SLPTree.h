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
        /// \param width The target width of the SLP vectors.
        explicit SLPTree(Operation* op, size_t width);

      private:

        void buildGraph(std::vector<Operation*> const& operations, SLPNode& parentNode);

        bool vectorizable(std::vector<Operation*> const& operations) const;
        bool commutative(std::vector<Operation*> const& operations) const;
        std::vector<Operation*> getOperands(std::vector<Operation*> const& values) const;
        std::vector<Operation*> getOperands(Operation* value) const;

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

        SLPTree::MODE modeFromOperation(Operation const* operation) const;

        std::vector<SLPNode> reorderOperands(SLPNode& multinode);

        std::vector<SLPNode> graphs;

        std::pair<SLPNode, SLPTree::MODE> getBest(SLPTree::MODE const& mode,
                                                  SLPNode const& last,
                                                  std::vector<SLPNode*> const& candidates) const;
      };
    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPGRAPH_H
