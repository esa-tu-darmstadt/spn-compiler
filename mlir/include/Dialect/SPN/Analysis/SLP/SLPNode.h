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

        explicit SLPNode(std::vector<Operation*> const& operations);

        void addOperationToLane(Operation* operation, size_t const& lane);
        std::vector<Operation*> getLastOperations() const;
        Operation* getOperation(size_t lane, size_t index);

        bool isMultiNode() const;
        bool areRootOfNode(std::vector<Operation*> const& operations) const;
        size_t numLanes() const;

        friend bool operator==(SLPNode const& lhs, SLPNode const& rhs) {
          return std::tie(lhs.lanes)
              == std::tie(rhs.lanes);
        }

        friend bool operator!=(SLPNode const& lhs, SLPNode const& rhs) {
          return !(lhs == rhs);
        }

      private:

        /// Stores lanes as lists of operations. An inner vector (i.e. a lane) only contains more than one operation
        /// if this node is a multinode.
        std::vector<std::vector<Operation*>> lanes;

      };

    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPNODE_H
