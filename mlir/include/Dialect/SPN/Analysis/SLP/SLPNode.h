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

        explicit SLPNode(std::vector<Operation*> const& values);

        void addOperands(std::vector<std::vector<Operation*>> const& operandsPerLane);
        void addOperands(std::vector<Operation*> const& operations, size_t const& lane);
        std::vector<SLPNode>& getOperands(size_t lane);

        std::vector<Operation*>& getLane(size_t lane);
        Operation* getOperation(size_t lane, size_t index);
        OperationName const& name();

        bool isMultiNode() const;
        size_t numLanes() const;
        bool attachable(std::vector<std::vector<Operation*>> const& operations);

        friend bool operator==(SLPNode const& lhs, SLPNode const& rhs) {
          return std::tie(lhs.operationName, lhs.lanes, lhs.operands)
              == std::tie(rhs.operationName, rhs.lanes, rhs.operands);
        }

        friend bool operator!=(SLPNode const& lhs, SLPNode const& rhs) {
          return !(lhs == rhs);
        }

      private:

        OperationName const operationName;

        /// Stores lanes as lists of operations. An inner vector (i.e. a lane) only contains more than one operation
        /// if this node is a multinode.
        std::vector<std::vector<Operation*>> lanes;

        /// A list of operands for each lane.
        std::vector<std::vector<SLPNode>> operands;

      };
    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPNODE_H
