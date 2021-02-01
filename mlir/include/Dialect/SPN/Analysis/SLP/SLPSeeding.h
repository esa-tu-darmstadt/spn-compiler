//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPSEEDING_H
#define SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPSEEDING_H

#include <SPN/Analysis/SPNNodeLevel.h>
#include "SPN/SPNOps.h"

namespace mlir {
  namespace spn {
    namespace slp {

      typedef std::vector<Operation*> seed_t;

      enum SearchMode {
        ///
        RootToLeaf,
        ///
        LeafToRoot,
        /// TODO?
        Chain
      };

      class SeedAnalysis {

      public:

        explicit SeedAnalysis(Operation* jointQuery);

        std::vector<seed_t> getSeeds(size_t const& op,
                                     SPNNodeLevel const& nodeLevels,
                                     SearchMode const& mode = RootToLeaf) const;

      private:

        Operation* jointQuery;

      };
    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPSEEDING_H
