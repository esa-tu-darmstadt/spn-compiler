//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H

#include "mlir/IR/Operation.h"
#include "SLPGraph.h"
#include "Seeding.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class SLPGraphBuilder {

        public:

          explicit SLPGraphBuilder(size_t maxLookAhead);

          std::shared_ptr<Superword> build(ArrayRef<Value> const& seed);

        private:

          enum Mode {
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

          void buildGraph(std::shared_ptr<Superword> const& superword);
          void reorderOperands(SLPNode* multinode) const;
          std::pair<Value, Mode> getBest(Mode const& mode, Value const& last, SmallVector<Value>& candidates) const;
          unsigned getLookAheadScore(Value const& last, Value const& candidate, unsigned maxLevel) const;

          // === Utilities === //

          Mode modeFromValue(Value const& value) const;
          std::shared_ptr<Superword> appendSuperwordToNode(ArrayRef<Value> const& values,
                                                           std::shared_ptr<SLPNode> const& node,
                                                           std::shared_ptr<Superword> const& usingSuperword);
          std::shared_ptr<SLPNode> addOperandToNode(ArrayRef<Value> const& operandValues,
                                                    std::shared_ptr<SLPNode> const& node,
                                                    std::shared_ptr<Superword> const& usingSuperword);
          std::shared_ptr<Superword> superwordOrNull(ArrayRef<Value> const& values) const;

          // ================= //

          size_t const maxLookAhead;
          DenseMap<Superword*, std::shared_ptr<SLPNode>> nodeBySuperword;
          DenseMap<Value, SmallVector<std::shared_ptr<Superword>>> superwordsByValue;
          SmallPtrSet<SLPNode*, 8> buildWorklist;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H
