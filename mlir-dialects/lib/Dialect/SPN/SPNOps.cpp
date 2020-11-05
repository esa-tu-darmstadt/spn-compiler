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
#include "mlir/IR/PatternMatch.h"
#include "Canonicalization/CanonicalizationPatterns.h"

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
      auto numWeights = op.weights().size();
      if (numWeights != numAddends) {
        return op.emitOpError("Number of weights must match the number of addends!");
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

namespace {

  template<typename NAryOp>
  unsigned numOperands(NAryOp& op) {
    return std::distance(op.operands().begin(), op.operands().end());
  }

}

//===----------------------------------------------------------------------===//
// ProductOp
//===----------------------------------------------------------------------===//

void mlir::spn::ProductOp::getCanonicalizationPatterns(::mlir::OwningRewritePatternList& results,
                                                       ::mlir::MLIRContext* context) {
  results.insert<ReduceProductOp>(context);
  results.insert<ConstantFoldProductOp>(context);
}

unsigned int mlir::spn::ProductOp::getNumOperands() {
  return numOperands<ProductOp>(*this);
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

void mlir::spn::SumOp::getCanonicalizationPatterns(::mlir::OwningRewritePatternList& results,
                                                   ::mlir::MLIRContext* context) {
  results.insert<ReduceSumOp>(context);
  results.insert<ConstantFoldSumOp>(context);
}

unsigned int mlir::spn::SumOp::getNumOperands() {
  return numOperands<SumOp>(*this);
}

//===----------------------------------------------------------------------===//
// WeightedSumOp
//===----------------------------------------------------------------------===//

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
        ArrayAttr::get(weightAttrs, odsBuilder.getContext()));
}

void mlir::spn::WeightedSumOp::getCanonicalizationPatterns(::mlir::OwningRewritePatternList& results,
                                                           ::mlir::MLIRContext* context) {
  results.insert<ReduceWeightedSumOp>(context);
  results.insert<ConstantFoldWeightedSumOp>(context);
}

unsigned int mlir::spn::WeightedSumOp::getNumOperands() {
  return numOperands<WeightedSumOp>(*this);
}

//===----------------------------------------------------------------------===//
// HistogramOp
//===----------------------------------------------------------------------===//

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

unsigned int mlir::spn::HistogramOp::getFeatureIndex() {
  if (auto blockArg = index().dyn_cast<BlockArgument>()) {
    return blockArg.getArgNumber();
  }
  // Expecting the index to be a block argument.
  assert(false);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

void mlir::spn::ReturnOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value retValue) {
  build(odsBuilder, odsState, ValueRange{retValue});
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void mlir::spn::ConstantOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, double value) {
  build(odsBuilder, odsState, ProbabilityType::get(odsBuilder.getContext()),
        odsBuilder.getFloatAttr(odsBuilder.getF64Type(), value));
}

//===----------------------------------------------------------------------===//
// SingleJointQuery
//===----------------------------------------------------------------------===//

std::vector<Operation*> mlir::spn::SingleJointQuery::getRootNodes() {
  // The graph (body region) has only a single block,
  // its terminator is the rootNode (result) of the graph.
  return {this->graph().front().getTerminator()};
}

unsigned int mlir::spn::SingleJointQuery::getNumFeatures() {
  return this->numFeatures();
}

#define GET_OP_CLASSES
#include "SPN/SPNOps.cpp.inc"