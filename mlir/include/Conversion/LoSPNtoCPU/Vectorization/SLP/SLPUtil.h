//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPUTIL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPUTIL_H

#include "mlir/IR/Operation.h"
#include "LoSPN/LoSPNOps.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        bool areConsecutiveLoads(low::SPNBatchRead load1, low::SPNBatchRead load2);
        bool areConsecutiveLoads(std::vector<Operation*> const& loads);

        template<typename OperationIterator>
        Operation* firstUser(OperationIterator begin, OperationIterator end) {
          Operation* firstUser = nullptr;
          while (begin != end) {
            for (auto* user : (*begin)->getUsers()) {
              if (!firstUser || user->isBeforeInBlock(firstUser)) {
                firstUser = user;
              }
            }
            ++begin;
          }
          return firstUser;
        }

        template<typename OperationIterator>
        Operation* firstOperation(OperationIterator begin, OperationIterator end) {
          Operation* firstOp = *begin;
          while (++begin != end) {
            if (!firstOp->isBeforeInBlock(*begin)) {
              firstOp = *begin;
            }
          }
          return firstOp;
        }

        template<typename OperationIterator>
        Operation* lastOperation(OperationIterator begin, OperationIterator end) {
          Operation* lastOp = *begin;
          while (++begin != end) {
            if (lastOp->isBeforeInBlock(*begin)) {
              lastOp = *begin;
            }
          }
          return lastOp;
        }

        template<typename ValueIterator>
        Value firstValue(ValueIterator begin, ValueIterator end) {
          Value firstVal = *begin;
          while (begin != end) {
            if (begin->template isa<BlockArgument>()) {
              return *begin;
            }
            if (!firstVal.getDefiningOp()->isBeforeInBlock(*begin)) {
              firstVal = *begin;
            }
            ++begin;
          }
          return firstVal;
        }

        template<typename ValueIterator>
        Value lastValue(ValueIterator begin, ValueIterator end) {
          Value lastVal = *begin;
          while (++begin != end) {
            if (begin->template isa<BlockArgument>()) {
              continue;
            } else if (lastVal.isa<BlockArgument>()
                || lastVal.getDefiningOp()->isBeforeInBlock(begin->getDefiningOp())) {
              lastVal = *begin;
            }
          }
          return lastVal;
        }

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPUTIL_H
