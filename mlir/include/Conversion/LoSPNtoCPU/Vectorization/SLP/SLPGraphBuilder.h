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

          std::unique_ptr<SLPNode> build(seed_t const& seed) const;

        private:

          void buildGraph(std::vector<Operation*> const& operations, SLPNode* currentNode) const;

          void reorderOperands(SLPNode* multinode) const;

          std::pair<Operation*, Mode> getBest(Mode const& mode,
                                              Operation* last,
                                              std::vector<Operation*>& candidates) const;

          int getLookAheadScore(Operation* last, Operation* candidate, size_t const& maxLevel) const;

          size_t const maxLookAhead;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPHBUILDER_H
