//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

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

          Value getElement(size_t lane) const;
          void setElement(size_t lane, Value const& value);
          Value operator[](size_t lane) const;
          bool contains(Value const& value) const;

          bool isLeaf() const;
          bool uniform() const;
          bool splattable() const;

          size_t numLanes() const;
          SmallVectorImpl<Value>::const_iterator begin() const;
          SmallVectorImpl<Value>::const_iterator end() const;

          size_t numOperands() const;
          void addOperand(ValueVector* operandVector);
          ValueVector* getOperand(size_t index) const;
          ArrayRef<ValueVector*> getOperands() const;

          std::shared_ptr<SLPNode> getParentNode() const;

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
          ArrayRef<std::shared_ptr<SLPNode>> getOperands() const;

        private:
          SmallVector<std::unique_ptr<ValueVector>> vectors;
          SmallVector<std::shared_ptr<SLPNode>> operandNodes;
        };

        namespace graph {

          template<typename Node>
          SmallVector<Node*> postOrder(Node* root) {
            SmallVector<Node*> order;
            // false = visit operands, true = insert into order
            SmallVector<std::pair<Node*, bool>> worklist;
            llvm::SmallSet<Node*, 32> finishedNodes;
            worklist.emplace_back(root, false);
            while (!worklist.empty()) {
              if (finishedNodes.contains(worklist.back().first)) {
                worklist.pop_back();
                continue;
              }
              auto* node = worklist.back().first;
              bool insert = worklist.back().second;
              worklist.pop_back();
              if (insert) {
                order.emplace_back(node);
                finishedNodes.insert(node);
              } else {
                worklist.emplace_back(node, true);
                for (size_t i = node->numOperands(); i-- > 0;) {
                  worklist.emplace_back(node->getOperand(i), false);
                }
              }
            }
            return order;
          }

        }
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H
