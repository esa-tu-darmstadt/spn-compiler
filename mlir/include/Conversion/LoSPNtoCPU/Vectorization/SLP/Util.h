//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_UTIL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_UTIL_H

#include "SLPGraph.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/IR/Operation.h"

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

        size_t numUniqueOps(ArrayRef<Superword*> const& superwords);

        void dumpSuperword(Superword const& superword);
        void dumpSLPNode(SLPNode const& node);

        void dumpOpGraph(ArrayRef<Value> const& values);
        void dumpSuperwordGraph(Superword* root);
        void dumpSLPGraph(SLPNode* root);

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_UTIL_H
