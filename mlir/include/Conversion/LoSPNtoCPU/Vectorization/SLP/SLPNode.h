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

          Operation* getOperation(size_t lane, size_t index) const;
          void setOperation(size_t lane, size_t index, Operation* operation);

          bool isMultiNode() const;
          bool isUniform() const;
          bool containsOperation(Operation* op) const;
          bool areRootOfNode(std::vector<Operation*> const& operations) const;

          size_t numLanes() const;

          size_t numVectors() const;
          void addVector(std::vector<Operation*> const& vectorOps);
          std::vector<Operation*>& getVector(size_t index);
          size_t getVectorIndex(Operation* op) const;
          std::vector<Operation*>& getVectorOf(Operation* op);
          std::vector<std::vector<Operation*>>& getVectors();

          SLPNode* addOperand(std::vector<Operation*> const& operations);
          SLPNode* getOperand(size_t index) const;
          std::vector<SLPNode*> getOperands() const;
          size_t numOperands() const;

          void addNodeInput(Value const& value);
          Value const& getNodeInput(size_t index) const;

          void dump() const;
          void dumpGraph() const;

          friend bool operator==(SLPNode const& lhs, SLPNode const& rhs) {
            return std::tie(lhs.vectors) == std::tie(rhs.vectors);
          }

          friend bool operator!=(SLPNode const& lhs, SLPNode const& rhs) {
            return !(lhs == rhs);
          }

        private:

          std::vector<std::vector<Operation*>> vectors;

          std::vector<std::unique_ptr<SLPNode>> operandNodes;

          std::vector<Value> nodeInputs;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPNODE_H
