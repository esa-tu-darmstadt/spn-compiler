//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "HiSPN/HiSPNOps.h"
#include "HiSPN/HiSPNDialect.h"
#include "HiSPN/HiSPNAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
  namespace spn {
    namespace high {

      namespace {

        template<typename HiSPNOp>
        unsigned numOperands(HiSPNOp op) {
          return std::distance(op.operands().begin(), op.operands().end());
        }

      }

      RegionKind mlir::spn::high::Graph::getRegionKind(unsigned int index) {
        return RegionKind::Graph;
      }

    } // end of namespace high
  } // end of namespace spn
} // end of namespace mlir

namespace {
  template<typename SourceOp>
  mlir::Operation* getParentQuery(SourceOp op) {
    // Parent should always be a Graph and the Graph's parent should always be
    // a Query, due to the HasParent-relationships defined in the operations.
    assert(op.getOperation()->template getParentOfType<mlir::spn::high::Graph>()
               && "Expecting the parent to be a Graph");
    return op.getOperation()->template getParentOfType<mlir::spn::high::Graph>()->getParentOp();
  }
}

//===----------------------------------------------------------------------===//
// ProductOp
//===----------------------------------------------------------------------===//

mlir::Operation* mlir::spn::high::ProductNode::getEnclosingQuery() {
  return getParentQuery(*this);
}

mlir::LogicalResult mlir::spn::high::ProductNode::verify() {
  auto numOps = getNumOperands();
  if (numOps == 0) {
    return emitOpError("Number of operands must be at least one");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

void mlir::spn::high::SumNode::build(::mlir::OpBuilder& odsBuilder,
                                     ::mlir::OperationState& odsState,
                                     llvm::ArrayRef<Value> operands,
                                     llvm::ArrayRef<double> weights) {
  SmallVector<mlir::Attribute, 10> weightAttrs;
  for (auto& w : weights) {
    weightAttrs.push_back(odsBuilder.getF64FloatAttr(w));
  }
  assert(weightAttrs.size() == operands.size() && "Number of weights must match number of operands!");
  build(odsBuilder, odsState, ProbabilityType::get(odsBuilder.getContext()), ValueRange(operands),
        ArrayAttr::get(odsBuilder.getContext(), weightAttrs));
}

mlir::Operation* mlir::spn::high::SumNode::getEnclosingQuery() {
  return getParentQuery(*this);
}

mlir::LogicalResult mlir::spn::high::SumNode::verify() {
  auto numAddends = getNumOperands();
  auto numWeights = getWeights().size();
  if (numWeights != numAddends) {
    return emitOpError("Number of weights must match the number of addends!");
  }
  if (numAddends <= 0) {
    return emitOpError("Number of addends must be greater than zero!");
  }
  double sum = 0.0;
  for (auto p : getWeights().getAsRange<mlir::FloatAttr>()) {
    sum += p.getValueAsDouble();
  }
  if (std::abs(sum - 1.0) > 1e-6) {
    return emitOpError("Weights must sum up to 1.0 for normalized SPN");
  }
  return mlir::success();
}



//===----------------------------------------------------------------------===//
// HistogramOp
//===----------------------------------------------------------------------===//

void mlir::spn::high::HistogramNode::build(::mlir::OpBuilder& odsBuilder,
                                           ::mlir::OperationState& odsState,
                                           Value indexVal,
                                           llvm::ArrayRef<std::tuple<int, int, APFloat>> buckets) {
  SmallVector<mlir::Attribute, 256> bucketList;
  // Create StructAttr for each bucket, comprising the inclusive lower bound,
  // the exclusive lower bound and the probability value.
  for (auto& bucket : buckets) {
    auto bucketAttr = high::HistBucketAttr::get(odsBuilder.getContext(), std::get<0>(bucket), 
                                                                        std::get<1>(bucket), 
                                                                        std::get<2>(bucket));
    bucketList.push_back(bucketAttr);
  }
  auto bucketListAttr = ArrayAttr::get(odsBuilder.getContext(), bucketList);
  build(odsBuilder, odsState, ProbabilityType::get(odsBuilder.getContext()), indexVal,
        bucketListAttr, odsBuilder.getUI32IntegerAttr(bucketList.size()));
}

unsigned int mlir::spn::high::HistogramNode::getFeatureIndex() {
  if (auto blockArg = getIndex().dyn_cast<BlockArgument>()) {
    return blockArg.getArgNumber();
  }
  // Expecting the index to be a block argument.
  assert(false);
}

mlir::Operation* mlir::spn::high::HistogramNode::getEnclosingQuery() {
  return getParentQuery(*this);
}

mlir::LogicalResult mlir::spn::high::HistogramNode::verify() {
  int64_t lb = std::numeric_limits<int64_t>::min();
  int64_t ub = std::numeric_limits<int64_t>::min();
  if (getBuckets().size() != getBucketCount()) {
    return emitOpError("bucketCount must match the actual number of buckets!");
  }
  for (auto bucket : getBuckets().getAsRange<high::HistBucketAttr>()) {
    auto curLB = bucket.getLowerBound();
    auto curUB = bucket.getUpperBound();
    if (curUB < curLB) {
      return emitOpError("Lower bound must be less or equal to upper bound!");
    }
    if (curLB > lb) {
      if (curLB < ub) {
        // The existing range and the new bucket overlap.
        return emitOpError("Overlapping buckets in histogram!");
      }
      ub = curUB;
    } else {
      if (curUB > lb) {
        // The new bucket and the existing range overlap.
        return emitOpError("Overlapping buckets in histogram!");
      }
      lb = curLB;
    }
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// CategoricalOp
//===----------------------------------------------------------------------===//

void mlir::spn::high::CategoricalNode::build(::mlir::OpBuilder& odsBuilder,
                                             ::mlir::OperationState& odsState,
                                             Value indexVal,
                                             llvm::ArrayRef<double> probabilities) {
  auto floatArrayAttr = odsBuilder.getF64ArrayAttr(probabilities);
  build(odsBuilder, odsState, ProbabilityType::get(odsBuilder.getContext()), indexVal, floatArrayAttr);
}

unsigned int mlir::spn::high::CategoricalNode::getFeatureIndex() {
  if (auto blockArg = getIndex().dyn_cast<BlockArgument>()) {
    return blockArg.getArgNumber();
  }
  // Expecting the index to be a block argument.
  assert(false);
}

mlir::Operation* mlir::spn::high::CategoricalNode::getEnclosingQuery() {
  return getParentQuery(*this);
}

mlir::LogicalResult mlir::spn::high::CategoricalNode::verify() {
  double sum = 0.0;
  for (auto p : getProbabilities().getAsRange<mlir::FloatAttr>()) {
    sum += p.getValueAsDouble();
  }
  if (std::abs(sum - 1.0) > 1e-6) {
    return emitOpError("Category probabilities should sum to 1.0");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GaussianOp
//===----------------------------------------------------------------------===//

void mlir::spn::high::GaussianNode::build(::mlir::OpBuilder& odsBuilder,
                                          ::mlir::OperationState& odsState,
                                          Value indexVal,
                                          double mean,
                                          double stddev) {
  build(odsBuilder, odsState, ProbabilityType::get(odsBuilder.getContext()), indexVal,
        odsBuilder.getF64FloatAttr(mean), odsBuilder.getF64FloatAttr(stddev));
}

unsigned int mlir::spn::high::GaussianNode::getFeatureIndex() {
  if (auto blockArg = getIndex().dyn_cast<BlockArgument>()) {
    return blockArg.getArgNumber();
  }
  // Expecting the index to be a block argument.
  assert(false);
}

mlir::Operation* mlir::spn::high::GaussianNode::getEnclosingQuery() {
  return getParentQuery(*this);
}

//===----------------------------------------------------------------------===//
// RootNode
//===----------------------------------------------------------------------===//

mlir::Operation* mlir::spn::high::RootNode::getEnclosingQuery() {
  return getParentQuery(*this);
}

//===----------------------------------------------------------------------===//
// SingleJointQuery
//===----------------------------------------------------------------------===//
mlir::LogicalResult mlir::spn::high::JointQuery::verify() {
  if (getSupportMarginal()) {
    // Marginalization is triggered by feature values set to NaN,
    // so the input must be a float type to represent that.
    if (!getFeatureDataType().isa<FloatType>()) {
      return emitOpError("Feature data type must be floating-point to support marginal");
    }
  }
  return mlir::success();

}

#define GET_TYPEDEF_CLASSES
#include "HiSPN/HiSPNOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "HiSPN/HiSPNOps.cpp.inc"