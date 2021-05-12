//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class ValueVector {

          friend class SLPNode;

        public:

          explicit ValueVector(ArrayRef<Value> const& values);
          explicit ValueVector(ArrayRef<Operation*> const& operations);

          bool contains(Value const& value) const;
          bool containsBlockArgs() const;
          bool splattable() const;
          bool isLeaf() const;

          size_t numLanes() const;
          size_t numOperands() const;
          ValueVector* getOperand(size_t index) const;

          SmallVectorImpl<Value>::const_iterator begin() const;
          SmallVectorImpl<Value>::const_iterator end() const;

          Value getElement(size_t lane) const;
          Value operator[](size_t lane) const;

        private:
          SmallVector<Value, 4> values;
          SmallVector<ValueVector*> operands;
        };

        class SLPNode {

        public:

          explicit SLPNode(ArrayRef<Value> const& values);
          explicit SLPNode(ArrayRef<Operation*> const& operations);

          Value getValue(size_t lane, size_t index) const;
          void setValue(size_t lane, size_t index, Value const& newValue);

          bool contains(Value const& value) const;

          bool isRootOfNode(ValueVector const& vector) const;

          size_t numLanes() const;
          size_t numVectors() const;

          ValueVector* addVector(ArrayRef<Value> const& values, ValueVector* definingVector);
          ValueVector* getVector(size_t index) const;
          ValueVector* getVectorOrNull(ArrayRef<Value> const& values) const;

          void addOperand(std::shared_ptr<SLPNode> operandNode, ValueVector* operandVector, ValueVector* definingVector);
          SLPNode* getOperand(size_t index) const;
          std::vector<SLPNode*> getOperands() const;
          size_t numOperands() const;

        private:
          SmallVector<std::unique_ptr<ValueVector>> vectors;
          SmallVector<std::shared_ptr<SLPNode>> operandNodes;
        };

        namespace graph {
          SmallVector<SLPNode*> postOrder(SLPNode* root);
        }

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H
