//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPNODE_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPNODE_H

#include "mlir/IR/Operation.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class SLPNode {

        public:

          explicit SLPNode(std::vector<Operation*> const& operations);

          void addOperationToLane(Operation* operation, size_t const& lane);
          Operation* getOperation(size_t lane, size_t index) const;
          void setOperation(size_t lane, size_t index, Operation* operation);

          bool isMultiNode() const;
          bool isUniform() const;
          bool areRootOfNode(std::vector<Operation*> const& operations) const;

          size_t numLanes() const;

          size_t numVectors() const;
          std::vector<Operation*> getVector(size_t index) const;

          SLPNode& addOperand(std::vector<Operation*> const& operations);
          SLPNode& getOperand(size_t index) const;
          std::vector<SLPNode*> getOperands() const;
          size_t numOperands() const;

          Type getResultType() const;

          void dump() const;

          friend bool operator==(SLPNode const& lhs, SLPNode const& rhs) {
            return std::tie(lhs.lanes) == std::tie(rhs.lanes);
          }

          friend bool operator!=(SLPNode const& lhs, SLPNode const& rhs) {
            return !(lhs == rhs);
          }

        private:

          /// Stores lanes as lists of operations. An inner vector (i.e. a lane) only contains more than one operation
          /// if this node is a multinode. Operations with smaller indices inside a lane are executed "after" the higher
          /// ones in the source code.
          std::vector<std::vector<Operation*>> lanes;

          std::vector<std::unique_ptr<SLPNode>> operandNodes;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPNODE_H
