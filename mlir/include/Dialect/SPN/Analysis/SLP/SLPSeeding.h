//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPSEEDING_H
#define SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPSEEDING_H

#include "SPN/SPNOps.h"

namespace mlir {
  namespace spn {
    namespace slp {
      namespace seeding {

        typedef std::vector<Operation*> seed_t;

        enum SearchMode {
          /// Look for disjoint subgraphs in the operation tree.
          DISJOINT,
          /// Look for the largest possible subgraph.
          SIZE,
          /// Stop looking for subgraphs.
          FAILED
        };

        std::vector<seed_t> getSeeds(Operation* root, size_t const& width);
      }
    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPSEEDING_H
