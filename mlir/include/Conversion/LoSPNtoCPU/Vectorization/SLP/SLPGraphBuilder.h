//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H

#include "mlir/IR/Operation.h"
#include "SLPNode.h"
#include "SLPSeeding.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class SLPGraphBuilder {

        public:

          explicit SLPGraphBuilder(size_t maxLookAhead);

          std::shared_ptr<SLPNode> build(ArrayRef<Value> const& seed);

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

          void buildGraph(NodeVector* vector, SLPNode* currentNode);
          void reorderOperands(SLPNode* multinode) const;
          std::pair<Value, Mode> getBest(Mode const& mode, Value const& last, SmallVector<Value>& candidates) const;
          int getLookAheadScore(Value const& last, Value const& candidate, unsigned maxLevel) const;

          static Mode modeFromValue(Value const& value);
          std::pair<std::shared_ptr<SLPNode>, size_t> getOrCreateOperand(ArrayRef<Value> values,
                                                                         bool* isNewOperand = nullptr);

          size_t const maxLookAhead;
          SmallVector<std::shared_ptr<SLPNode>> nodes;
          SmallPtrSet<SLPNode*, 8> reorderWorklist;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H
