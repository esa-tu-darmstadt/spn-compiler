//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H

#include "mlir/IR/Operation.h"
#include "SLPMode.h"
#include "SLPNode.h"
#include "SLPSeeding.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class SLPGraphBuilder {

        public:

          explicit SLPGraphBuilder(size_t maxLookAhead);

          std::unique_ptr<SLPNode> build(vector_t const& seed) const;

        private:

          void buildGraph(NodeVector* vector, SLPNode* currentNode) const;

          void reorderOperands(SLPNode* multinode) const;

          std::pair<Value, Mode> getBest(Mode const& mode, Value const& last, SmallVector<Value>& candidates) const;

          int getLookAheadScore(Value const& last, Value const& candidate, unsigned maxLevel) const;

          size_t const maxLookAhead;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H
