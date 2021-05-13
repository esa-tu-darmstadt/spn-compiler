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

        class SLPNode;

        class ValueVector {

          friend SLPNode;

        public:

          ValueVector(ArrayRef<Value> const& values, std::shared_ptr<SLPNode> const& parentNode);
          ValueVector(ArrayRef<Operation*> const& operations, std::shared_ptr<SLPNode> const& parentNode);

          bool contains(Value const& value) const;
          bool containsBlockArgs() const;
          bool splattable() const;
          bool isLeaf() const;

          size_t numLanes() const;
          size_t numOperands() const;

          void addOperand(ValueVector* operandVector);
          ValueVector* getOperand(size_t index) const;
          std::shared_ptr<SLPNode> getParentNode() const;

          SmallVectorImpl<Value>::const_iterator begin() const;
          SmallVectorImpl<Value>::const_iterator end() const;

          Value getElement(size_t lane) const;
          Value operator[](size_t lane) const;

        private:
          SmallVector<Value, 4> values;
          SmallVector<ValueVector*> operands;
          std::weak_ptr<SLPNode> const parentNode;
        };

        class SLPNode {

        public:

          ValueVector* addVector(std::unique_ptr<ValueVector> vector);
          ValueVector* getVector(size_t index) const;

          Value getValue(size_t lane, size_t index) const;
          void setValue(size_t lane, size_t index, Value const& newValue);

          bool contains(Value const& value) const;

          bool isVectorRoot(ValueVector const& vector) const;

          size_t numLanes() const;
          size_t numVectors() const;
          size_t numOperands() const;

          void addOperand(std::shared_ptr<SLPNode> operandNode);
          SLPNode* getOperand(size_t index) const;
          std::vector<SLPNode*> getOperands() const;

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
