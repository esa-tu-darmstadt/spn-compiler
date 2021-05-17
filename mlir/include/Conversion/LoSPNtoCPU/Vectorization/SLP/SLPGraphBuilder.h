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

          std::shared_ptr<ValueVector> build(ArrayRef<Value> const& seed);

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

          void buildGraph(std::shared_ptr<ValueVector> const& vector);
          void reorderOperands(SLPNode* multinode) const;
          std::pair<Value, Mode> getBest(Mode const& mode, Value const& last, SmallVector<Value>& candidates) const;
          unsigned getLookAheadScore(Value const& last, Value const& candidate, unsigned maxLevel) const;

          // === Utilities === //

          Mode modeFromValue(Value const& value) const;
          std::shared_ptr<ValueVector> appendVectorToNode(ArrayRef<Value> const& values,
                                                          std::shared_ptr<SLPNode> const& node,
                                                          std::shared_ptr<ValueVector> const& usingVector);
          std::shared_ptr<SLPNode> addOperandToNode(ArrayRef<Value> const& operandValues,
                                                    std::shared_ptr<SLPNode> const& node,
                                                    std::shared_ptr<ValueVector> const& usingVector);
          std::shared_ptr<ValueVector> vectorOrNull(ArrayRef<Value> const& values) const;

          // ================= //

          size_t const maxLookAhead;
          DenseMap<ValueVector*, std::shared_ptr<SLPNode>> nodesByVector;
          DenseMap<Value, SmallVector<std::shared_ptr<ValueVector>>> vectorsByValue;
          SmallPtrSet<SLPNode*, 8> buildWorklist;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H
