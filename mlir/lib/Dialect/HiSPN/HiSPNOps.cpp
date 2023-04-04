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

namespace mlir {
  namespace spn {
    namespace high {

      namespace {

        template<typename HiSPNOp>
        unsigned numOperands(HiSPNOp op) {
          return std::distance(op.getOperands().begin(), op.getOperands().end());
        }

      }

      //===----------------------------------------------------------------------===//
      // JointQuery
      //===----------------------------------------------------------------------===//
      mlir::LogicalResult JointQuery::verify() {
        if (getSupportMarginal()) {
          // Marginalization is triggered by feature values set to NaN,
          // so the input must be a float type to represent that.
          if (!getFeatureDataType().isa<FloatType>()) {
            return emitOpError("Feature data type must be floating-point to support marginal");
          }
        }
        return mlir::success();
      }

      //===----------------------------------------------------------------------===//
      // ProductNode
      //===----------------------------------------------------------------------===//
      mlir::LogicalResult ProductNode::verify() {
        auto numOps = numOperands(*this);
        if (numOps == 0) {
          return emitOpError("Number of operands must be at least one");
        }
        return mlir::success();
      }

      //===----------------------------------------------------------------------===//
      // SumNode
      //===----------------------------------------------------------------------===//

      mlir::LogicalResult SumNode::verify() {
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

      mlir::LogicalResult HistogramNode::verify() {
        int64_t lb = std::numeric_limits<int64_t>::min();
        int64_t ub = std::numeric_limits<int64_t>::min();
        if (getBuckets().size() != getBucketCount()) {
          return emitOpError("bucketCount must match the actual number of buckets!");
        }
        auto buckets = this->getBuckets();
        for (auto b : buckets.getValue()) {
          auto bucket = b.cast<BucketAttr>();
          auto curLB = bucket.getLb();
          auto curUB = bucket.getUb();
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

      mlir::LogicalResult CategoricalNode::verify() {
        double sum = 0.0;
        for (auto p : getProbabilities().getAsRange<mlir::FloatAttr>()) {
          sum += p.getValueAsDouble();
        }
        if (std::abs(sum - 1.0) > 1e-6) {
          return emitOpError("Category probabilities should sum to 1.0");
        }
        return mlir::success();
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


//===----------------------------------------------------------------------===//
// HistogramOp
//===----------------------------------------------------------------------===//

void mlir::spn::high::HistogramNode::build(::mlir::OpBuilder& odsBuilder,
                                           ::mlir::OperationState& odsState,
                                           Value indexVal,
                                           llvm::ArrayRef<std::tuple<int, int, double>> buckets) {
  SmallVector<mlir::Attribute, 256> bucketList;
  // Create StructAttr for each bucket, comprising the inclusive lower bound,
  // the exclusive upper bound and the probability value.
  for (auto& bucket : buckets) {
    int lb = std::get<0>(bucket);
    int ub = std::get<1>(bucket);
    double prob = std::get<2>(bucket);

    auto bucketAttr = BucketAttr::get(
      odsBuilder.getContext(), lb, ub, APFloat(prob)
    );

    bucketList.push_back(bucketAttr);
  }

  auto arrAttr = odsBuilder.getArrayAttr(bucketList);
  build(odsBuilder, odsState, ProbabilityType::get(odsBuilder.getContext()), indexVal,
        arrAttr, odsBuilder.getUI32IntegerAttr(bucketList.size()));
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

//unsigned int mlir::spn::high::JointQuery::getNumFeatures() {
//  return this->getNumFeatures();
//}

mlir::Type mlir::spn::high::JointQuery::getFeatureDataType() {
  return this->getInputType();
}

//unsigned int mlir::spn::high::JointQuery::getBatchSize() {
//  return this->getBatchSize();
//}

//mlir::spn::high::error_model mlir::spn::high::JointQuery::getErrorModel() {
//  return this->getErrorModel();
//}

//::llvm::APFloat mlir::spn::high::JointQuery::getMaxError() {
//  return this->getMaxError();
//}

llvm::StringRef mlir::spn::high::JointQuery::getQueryName() {
  return this->getKernelName();
}

#define GET_OP_CLASSES
#include "HiSPN/HiSPNOps.cpp.inc"