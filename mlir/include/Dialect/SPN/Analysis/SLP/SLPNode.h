//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPNODE_H
#define SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPNODE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
  namespace spn {
    namespace slp {

      class SLPNode {

      public:

        explicit SLPNode(size_t const& width);

        explicit SLPNode(std::vector<Operation*> const& values);

      private:

        size_t const width;

        /// List of operations that can be executed in any order/in parallel when width > 1.
        std::vector<Operation*> operations;
        std::vector<SLPNode> operands;

      };
    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPNODE_H
