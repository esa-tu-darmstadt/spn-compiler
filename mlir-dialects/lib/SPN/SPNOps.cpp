//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/SPNOps.h"
#include "SPN/SPNDialect.h"
#include "SPN/SPNAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::spn;

namespace mlir {
  namespace spn {
    //===----------------------------------------------------------------------===//
    // N-ary operations.
    //===----------------------------------------------------------------------===//

    template<typename NAryOp>
    static mlir::LogicalResult verify(NAryOp op) {
      auto numOperands = std::distance(op.operands().begin(), op.operands().end());
      if (numOperands != op.opCount()) {
        return op.emitOpError("Number of operands must match the specified operand count!");
      }
      if (numOperands <= 0) {
        return op.emitOpError("Number of operands must be greater than zero!");
      }
      return mlir::success();
    }


    //===----------------------------------------------------------------------===//
    // ProductOp
    //===----------------------------------------------------------------------===//

    template mlir::LogicalResult verify<ProductOp>(ProductOp op);

    //===----------------------------------------------------------------------===//
    // SumOp
    //===----------------------------------------------------------------------===//

    template mlir::LogicalResult verify<SumOp>(SumOp op);

    //===----------------------------------------------------------------------===//
    // WeightedSumOp
    //===----------------------------------------------------------------------===//

    static mlir::LogicalResult verify(WeightedSumOp op) {
      auto numAddends = std::distance(op.operands().begin(), op.operands().end());
      if (numAddends != op.opCount()) {
        return op.emitOpError("Number of addends must match the specified operand count!");
      }
      auto numWeights = op.weights().size();
      if (numWeights != op.opCount()) {
        return op.emitOpError("Number of weights must match the specified operand count!");
      }
      if (numAddends <= 0) {
        return op.emitOpError("Number of addends must be greater than zero!");
      }
      return mlir::success();
    }

    //===----------------------------------------------------------------------===//
    // HistogramOp
    //===----------------------------------------------------------------------===//

    static mlir::LogicalResult verify(HistogramOp op) {
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

    RegionKind mlir::spn::SingleJointQuery::getRegionKind(unsigned int index) {
      return RegionKind::Graph;
    }

  } // end of namespace spn
} // end of namespace mlir

void mlir::spn::WeightedSumOp::build(::mlir::OpBuilder& odsBuilder,
                                     ::mlir::OperationState& odsState,
                                     llvm::ArrayRef<Value> operands,
                                     llvm::ArrayRef<double> weights) {
  SmallVector<mlir::Attribute, 10> weightAttrs;
  for (auto& w : weights) {
    weightAttrs.push_back(odsBuilder.getF64FloatAttr(w));
  }
  assert(weightAttrs.size() == operands.size() && "Number of weights must match number of operands!");
  build(odsBuilder, odsState, ProbabilityType::get(odsBuilder.getContext()), ValueRange(operands),
        ArrayAttr::get(weightAttrs, odsBuilder.getContext()), odsBuilder.getUI32IntegerAttr(operands.size()));
}

void mlir::spn::HistogramOp::build(::mlir::OpBuilder& odsBuilder,
                                   ::mlir::OperationState& odsState,
                                   Value indexVal,
                                   llvm::ArrayRef<std::tuple<int, int, double>> buckets) {
  SmallVector<mlir::Attribute, 256> bucketList;
  // Create StructAttr for each bucket, comprising the inclusive lower bound,
  // the exclusive lower bound and the probability value.
  for (auto& bucket : buckets) {
    auto bucketAttr = Bucket::get(odsBuilder.getI32IntegerAttr(std::get<0>(bucket)),
                                  odsBuilder.getI32IntegerAttr(std::get<1>(bucket)),
                                  odsBuilder.getF64FloatAttr(std::get<2>(bucket)), odsBuilder.getContext());
    bucketList.push_back(bucketAttr);
  }
  auto arrAttr = odsBuilder.getArrayAttr(bucketList);
  build(odsBuilder, odsState, ProbabilityType::get(odsBuilder.getContext()), indexVal,
        arrAttr, odsBuilder.getUI32IntegerAttr(bucketList.size()));
}

void mlir::spn::ReturnOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value retValue) {
  build(odsBuilder, odsState, ValueRange{retValue});
}

#define GET_OP_CLASSES
#include "SPN/SPNOps.cpp.inc"


