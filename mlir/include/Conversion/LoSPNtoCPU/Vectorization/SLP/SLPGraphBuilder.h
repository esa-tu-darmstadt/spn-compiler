//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H

#include "SLPGraph.h"
#include "ScoreModel.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/IR/Operation.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class SLPGraphBuilder {

        public:

          explicit SLPGraphBuilder(SLPGraph& graph);

          void build(ArrayRef<Value> seed);

        private:

          enum Mode {
            /// look for a constant
            CONST,
            /// look for a consecutive load to that in the previous lane
            LOAD,
            /// look for an operation of the same opcode
            OPCODE,
            /// look for the exact same operation
            SPLAT,
            /// vectorization has failed, give higher priority to others
            FAILED
          };

          void buildGraph(std::shared_ptr<Superword> const& superword);
          void reorderOperands(SLPNode* multinode) const;
          std::pair<Value, Mode> getBest(Mode mode, Value last, SmallVector<Value>& candidates) const;

          // === Utilities === //

          static Mode modeFromValue(Value value);

          std::shared_ptr<Superword> appendSuperwordToNode(ArrayRef<Value> values,
                                                           std::shared_ptr<SLPNode> const& node,
                                                           std::shared_ptr<Superword> const& usingSuperword);
          std::shared_ptr<SLPNode> addOperandToNode(ArrayRef<Value> operandValues,
                                                    std::shared_ptr<SLPNode> const& node,
                                                    std::shared_ptr<Superword> const& usingSuperword);
          std::shared_ptr<Superword> superwordOrNull(ArrayRef<Value> values) const;

          // ================= //

          SLPGraph& graph;
          /// The model to use for look-ahead computation.
          std::unique_ptr<ScoreModel> scoreModel;
          DenseMap<Superword*, std::shared_ptr<SLPNode>> nodeBySuperword;
          DenseMap<Value, SmallVector<std::shared_ptr<Superword>>> superwordsByValue;
          /// Prevents building nodes more than once in case they appear several times in the graph.
          SmallPtrSet<SLPNode*, 8> buildWorklist;
          /// Prevents building superwords that are topologically mixed.
          DenseMap<Value, unsigned> valueDepths;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H
