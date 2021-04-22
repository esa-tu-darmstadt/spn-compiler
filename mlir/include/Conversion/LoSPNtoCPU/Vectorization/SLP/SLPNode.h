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

        typedef SmallVector<Value, 4> vector_t;

        class NodeVector {

          friend class SLPNode;

        public:

          explicit NodeVector(vector_t const& values);
          explicit NodeVector(SmallVector<Operation*, 4> const& operations);

          bool isUniform() const;
          bool contains(Value const& value) const;
          bool containsBlockArgs() const;
          bool vectorizable() const;

          size_t numLanes() const;

          size_t numOperands() const;
          NodeVector* getOperand(size_t index) const;

          vector_t::const_iterator begin() const;
          vector_t::const_iterator end() const;

          Value const& getElement(size_t lane) const;
          Value const& operator[](size_t lane) const;

        private:
          vector_t values;
          SmallVector<std::shared_ptr<NodeVector>> operands;
        };

        class SLPNode {

        public:

          explicit SLPNode(vector_t const& values);
          explicit SLPNode(SmallVector<Operation*, 4> const& operations);

          Value getValue(size_t lane, size_t index) const;
          void setValue(size_t lane, size_t index, Value const& newValue);

          bool isUniform() const;
          bool contains(Value const& value) const;

          bool isRootOfNode(NodeVector const& vector) const;

          size_t numLanes() const;
          size_t numVectors() const;

          NodeVector* addVector(vector_t const& values, NodeVector* definingVector);
          NodeVector* addVector(SmallVector<Operation*, 4> const& operations);
          NodeVector* getVector(size_t index) const;

          SLPNode* addOperand(vector_t const& values, NodeVector* definingVector);
          SLPNode* getOperand(size_t index) const;
          std::vector<SLPNode*> getOperands() const;
          size_t numOperands() const;

          static SmallVector<SLPNode*> postOrder(SLPNode* root);
          static DenseMap<NodeVector*, SmallVector<size_t, 4>> escapingLanesMap(SLPNode* root);

/*
          bool operator==(SLPNode const& other) {
            return std::tie(vectors) == std::tie(other.vectors);
          }

          bool operator!=(SLPNode const& other) {
            return !(*this == other);
          }
*/

        private:
          SmallVector<std::shared_ptr<NodeVector>> vectors;
          SmallVector<std::unique_ptr<SLPNode>> operandNodes;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPNODE_H
