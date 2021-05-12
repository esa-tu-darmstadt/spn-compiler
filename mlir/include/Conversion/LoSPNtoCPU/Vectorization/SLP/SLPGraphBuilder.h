//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

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

          void buildGraph(ValueVector* vector, SLPNode* currentNode);
          void reorderOperands(SLPNode* multinode) const;
          std::pair<Value, Mode> getBest(Mode const& mode, Value const& last, SmallVector<Value>& candidates) const;
          unsigned getLookAheadScore(Value const& last, Value const& candidate, unsigned maxLevel) const;

          static Mode modeFromValue(Value const& value);
          Optional<std::pair<size_t, ValueVector*>> nodeOrNone(ArrayRef<Value> const& values) const;

          size_t const maxLookAhead;
          SmallVector<std::shared_ptr<SLPNode>> nodes;
          SmallPtrSet<SLPNode*, 8> buildWorklist;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H
