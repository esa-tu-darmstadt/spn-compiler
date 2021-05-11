//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_UTIL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_UTIL_H

#include "SLPGraph.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Vector/VectorOps.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        bool vectorizable(Operation* op);
        bool vectorizable(Value const& value);

        template<typename ValueIterator>
        bool vectorizable(ValueIterator begin, ValueIterator end) {
          if (begin->template isa<BlockArgument>()) {
            return false;
          }
          auto const& name = begin->getDefiningOp()->getName();
          ++begin;
          while (begin != end) {
            if (!vectorizable(*begin) || begin->getDefiningOp()->getName() != name) {
              return false;
            }
            ++begin;
          }
          return true;
        }

        bool ofVectorizableType(Value const& value);

        template<typename ValueIterator>
        bool ofVectorizableType(ValueIterator begin, ValueIterator end) {
          return std::all_of(begin, end, [&](auto const& value) {
            return ofVectorizableType(value);
          });
        }

        template<typename ValueIterator>
        bool commutative(ValueIterator begin, ValueIterator end) {
          while (begin != end) {
            if (begin->template isa<BlockArgument>()
                || !begin->getDefiningOp()->template hasTrait<OpTrait::IsCommutative>()) {
              return false;
            }
            ++begin;
          }
          return true;
        }

        bool consecutiveLoads(Value const& lhs, Value const& rhs);

        template<typename ValueIterator>
        bool consecutiveLoads(ValueIterator begin, ValueIterator end) {
          Value previous = *begin;
          if (++begin == end || previous.isa<BlockArgument>() || !dyn_cast<SPNBatchRead>(previous.getDefiningOp())) {
            return false;
          }
          while (begin != end) {
            Value current = *begin;
            if (!consecutiveLoads(previous, current)) {
              return false;
            }
            previous = current;
            ++begin;
          }
          return true;
        }

        template<typename ValueIterator>
        Value broadcastFirstInsertRest(ValueIterator begin,
                                       ValueIterator end,
                                       VectorType const& vectorType,
                                       PatternRewriter& rewriter) {
          Value vectorOp = rewriter.create<vector::BroadcastOp>(begin->getLoc(), vectorType, *begin);
          unsigned position = 1;
          while (++begin != end) {
            vectorOp = rewriter.create<vector::InsertElementOp>(begin->getLoc(), *begin, vectorOp, position++);
          }
          return vectorOp;
        }

        size_t numNodes(SLPNode* root);
        size_t numVectors(SLPNode* root);

        void dumpOpTree(ArrayRef<Value> const& values);
        void dumpSLPGraph(SLPNode* root);
        void dumpSLPNode(SLPNode const& node);
        void dumpSLPNodeVector(NodeVector const& nodeVector);

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_UTIL_H
