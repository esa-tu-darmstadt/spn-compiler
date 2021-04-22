//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPUTIL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPUTIL_H

#include "SLPNode.h"
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

        bool isBeforeInBlock(Operation* lhs, Operation* rhs);
        bool isBeforeInBlock(Value const& lhs, Value const& rhs);

        bool consecutiveLoads(Value const& lhs, Value const& rhs);

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

        template<typename Iterator>
        typename std::iterator_traits<Iterator>::value_type firstOccurrence(Iterator begin, Iterator end) {
          typename std::iterator_traits<Iterator>::value_type first = *begin;
          while (++begin != end) {
            if (!isBeforeInBlock(first, *begin)) {
              first = *begin;
            }
          }
          return first;
        }

        template<typename Iterator>
        typename std::iterator_traits<Iterator>::value_type lastOccurrence(Iterator begin, Iterator end) {
          typename std::iterator_traits<Iterator>::value_type last = *begin;
          while (++begin != end) {
            if (isBeforeInBlock(last, *begin)) {
              last = *begin;
            }
          }
          return last;
        }

        template<typename UsableIterator>
        Operation* firstUser(UsableIterator begin, UsableIterator end) {
          Operation* firstUser = nullptr;
          while (begin != end) {
            for (auto* user : begin->getUsers()) {
              if (!firstUser || user->isBeforeInBlock(firstUser)) {
                firstUser = user;
              }
            }
            ++begin;
          }
          return firstUser;
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

        void dumpOpTree(vector_t const& values);
        void dumpSLPGraph(SLPNode const& root);
        void dumpSLPNode(SLPNode const& node);
        void dumpSLPNodeVector(NodeVector const& nodeVector);

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPUTIL_H
