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
#include "SPN/SPNOpTraits.h"
#include "SLPMode.h"
#include "SLPNode.h"
#include "SLPSeeding.h"

#include <vector>
#include <set>

namespace mlir {
  namespace spn {
    namespace slp {

      class SLPTree {

      public:

        explicit SLPTree(seed_t const& seed, size_t maxLookAhead);

      private:

        void buildGraph(std::vector<Operation*> const& operations, SLPNode& parentNode);

        /// Checks if the given operations are vectorizable. Operations are vectorizable iff the SPN dialect says they're
        /// vectorizable and they all share the same opcode.
        /// \param operations The potentially vectorizable operations.
        /// \return True if the operations can be vectorized, otherwise false.
        static bool vectorizable(std::vector<Operation*> const& operations) {
          for (size_t i = 0; i < operations.size(); ++i) {
            if (!operations.at(i)->hasTrait<OpTrait::spn::Vectorizable>()
                || (i > 0 && operations.at(i)->getName() != operations.front()->getName())) {
              return false;
            }
          }
          return true;
        }

        static bool commutative(std::vector<Operation*> const& operations) {
          return std::all_of(std::begin(operations), std::end(operations), [&](Operation* operation) {
            return operation->hasTrait<OpTrait::IsCommutative>();
          });
        }

        static std::vector<Operation*> getOperands(std::vector<Operation*> const& values) {
          std::vector<Operation*> operands;
          for (auto const& operation : values) {
            for (auto operand : operation->getOperands()) {
              operands.emplace_back(operand.getDefiningOp());
            }
          }
          return operands;
        }

        static std::vector<Operation*> getOperands(Operation* operation) {
          std::vector<Operation*> operands;
          for (auto operand : operation->getOperands()) {
            operands.emplace_back(operand.getDefiningOp());
          }
          return operands;
        }

        std::vector<std::vector<SLPNode>> reorderOperands(SLPNode& multinode);

        std::vector<SLPNode> graphs;

        size_t const maxLookAhead;

        std::pair<Optional<SLPNode>, Mode> getBest(Mode const& mode,
                                                   SLPNode const& last,
                                                   std::vector<SLPNode>& candidates) const;
        int getLookAheadScore(SLPNode const& last, SLPNode const& candidate, size_t const& maxLevel) const;
      };
    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPGRAPH_H
