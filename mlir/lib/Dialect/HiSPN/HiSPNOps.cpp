//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "HiSPN/HiSPNOps.h"
#include "HiSPN/HiSPNDialect.h"
#include "HiSPN/HiSPNAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
  namespace spn {
    namespace high {

      namespace {

        template<typename HiSPNOp>
        unsigned numOperands(HiSPNOp op) {
          return std::distance(op.operands().begin(), op.operands().end());
        }

      }

      //===----------------------------------------------------------------------===//
      // ProductNode
      //===----------------------------------------------------------------------===//
      static mlir::LogicalResult verify(ProductNode node) {
        auto numOps = numOperands(node);
        if (numOps == 0) {
          return node->emitOpError("Number of operands must be at least one");
        }
        return mlir::success();
      }

      //===----------------------------------------------------------------------===//
      // WeightedSumNode
      //===----------------------------------------------------------------------===//

      static mlir::LogicalResult verify(SumNode op) {
        auto numAddends = numOperands(op);
        auto numWeights = op.weights().size();
        if (numWeights != numAddends) {
          return op.emitOpError("Number of weights must match the number of addends!");
        }
        if (numAddends <= 0) {
          return op.emitOpError("Number of addends must be greater than zero!");
        }
        double sum = 0.0;
        for (auto p : op.weights().getAsRange<mlir::FloatAttr>()) {
          sum += p.getValueAsDouble();
        }
        if (std::abs(sum - 1.0) > 1e-6) {
          return op.emitOpError("Weights must sum up to 1.0 for normalized SPN");
        }
        return mlir::success();
      }

      //===----------------------------------------------------------------------===//
      // HistogramOp
      //===----------------------------------------------------------------------===//

      static mlir::LogicalResult verify(HistogramNode op) {
        int64_t lb = std::numeric_limits<int64_t>::min();
        int64_t ub = std::numeric_limits<int64_t>::min();
        if (op.buckets().size() != op.bucketCount()) {
          return op.emitOpError("bucketCount must match the actual number of buckets!");
        }
        auto buckets = op.buckets();
        for (auto b : buckets.getValue()) {
          auto bucket = b.cast<DictionaryAttr>();
          auto curLB = bucket.get("lb").cast<IntegerAttr>().getInt();
          auto curUB = bucket.get("ub").cast<IntegerAttr>().getInt();
          if (curUB < curLB) {
            return op.emitOpError("Lower bound must be less or equal to upper bound!");
          }
          if (curLB > lb) {
            if (curLB < ub) {
              // The existing range and the new bucket overlap.
              return op.emitOpError("Overlapping buckets in histogram!");
            }
            ub = curUB;
          } else {
            if (curUB > lb) {
              // The new bucket and the existing range overlap.
              return op.emitOpError("Overlapping buckets in histogram!");
            }
            lb = curLB;
          }
        }
        return mlir::success();
      }

      //===----------------------------------------------------------------------===//
      // CategoricalOp
      //===----------------------------------------------------------------------===//

      static mlir::LogicalResult verify(CategoricalNode op) {
        double sum = 0.0;
        for (auto p : op.probabilities().getAsRange<mlir::FloatAttr>()) {
          sum += p.getValueAsDouble();
        }
        if (std::abs(sum - 1.0) > 1e-6) {
          return op.emitOpError("Category probabilities should sum to 1.0");
        }
        return mlir::success();
      }

      RegionKind mlir::spn::high::Graph::getRegionKind(unsigned int index) {
        return RegionKind::Graph;
      }

    } // end of namespace high
  } // end of namespace spn
} // end of namespace mlir

#define GET_OP_CLASSES
#include "HiSPN/HiSPNOps.cpp.inc"