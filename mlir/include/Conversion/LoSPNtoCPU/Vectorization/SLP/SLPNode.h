//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPNODE_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPNODE_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class NodeVector {

          friend class SLPNode;

        public:

          explicit NodeVector(ArrayRef<Value> const& values);
          explicit NodeVector(ArrayRef<Operation*> const& operations);

          bool contains(Value const& value) const;
          bool containsBlockArgs() const;
          bool splattable() const;
          bool isLeaf() const;

          size_t numLanes() const;
          size_t numOperands() const;
          NodeVector* getOperand(size_t index) const;

          SmallVectorImpl<Value>::const_iterator begin() const;
          SmallVectorImpl<Value>::const_iterator end() const;

          Value const& getElement(size_t lane) const;
          Value const& operator[](size_t lane) const;

        private:
          SmallVector<Value, 4> values;
          SmallVector<NodeVector*> operands;
        };

        class SLPNode {

        public:

          explicit SLPNode(ArrayRef<Value> const& values);
          explicit SLPNode(ArrayRef<Operation*> const& operations);

          Value getValue(size_t lane, size_t index) const;
          void setValue(size_t lane, size_t index, Value const& newValue);

          bool contains(Value const& value) const;

          bool isRootOfNode(NodeVector const& vector) const;

          size_t numLanes() const;
          size_t numVectors() const;

          NodeVector* addVector(ArrayRef<Value> const& values, NodeVector* definingVector);
          NodeVector* getVector(size_t index) const;
          NodeVector* getVectorOrNull(ArrayRef<Value> const& values) const;

          void addOperand(std::shared_ptr<SLPNode> operandNode, NodeVector* operandVector, NodeVector* definingVector);
          SLPNode* getOperand(size_t index) const;
          std::vector<SLPNode*> getOperands() const;
          size_t numOperands() const;

          static SmallVector<SLPNode const*> postOrder(SLPNode const& root);

        private:
          SmallVector<std::unique_ptr<NodeVector>> vectors;
          SmallVector<std::shared_ptr<SLPNode>> operandNodes;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPNODE_H
