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

        SLPNode& addOperands(std::vector<Operation*> const& values);
        std::vector<SLPNode>& getOperands();
        SLPNode& getOperand(size_t index);

        std::vector<Operation*> getLane(size_t index);
        Operation* getOperation(size_t lane, size_t index);
        OperationName const& name();

        bool isMultiNode() const;
        size_t numLanes() const;
        bool attachable(std::vector<Operation*> const& otherOperations);

        friend bool operator==(SLPNode const& lhs, SLPNode const& rhs) {
          return std::tie(lhs.width, lhs.operationName, lhs.lanes, lhs.operands)
              == std::tie(rhs.width, rhs.operationName, rhs.lanes, rhs.operands);
        }

        friend bool operator!=(SLPNode const& lhs, SLPNode const& rhs) {
          return !(lhs == rhs);
        }

      private:

        size_t const width;
        OperationName const operationName;

        /// Stores lists of operations for each lane for multinode use cases.
        std::vector<std::vector<Operation*>> lanes;
        std::vector<SLPNode> operands;

      };
    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPNODE_H
