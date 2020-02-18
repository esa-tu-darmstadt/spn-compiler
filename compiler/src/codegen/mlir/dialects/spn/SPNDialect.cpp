//
// Created by ls on 2/7/20.
//

#include "SPNDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::spn;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
SPNDialect::SPNDialect(mlir::MLIRContext *ctx) : mlir::Dialect("spn", ctx) {
  addOperations<
#define GET_OP_LIST
#include "src/codegen/mlir/dialects/spn/SPNOps.cpp.inc"
  >();

  addAttributes<spn::Bucket>();
}

//===----------------------------------------------------------------------===//
// ConstantOp

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::Builder *builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder->getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.
static mlir::LogicalResult verify(ConstantOp op) {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType = op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = op.value().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank()) {
    return op.emitOpError(
            "return type must match the one of the attached value "
            "attribute: ")
            << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return op.emitOpError(
          "return type shape mismatches its attribute at dimension ")
          << dim << ": " << attrType.getShape()[dim]
          << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

static mlir::LogicalResult verify(ProductOp op) {
  auto numMultiplicands = std::distance(op.multiplicands().begin(), op.multiplicands().end());
  if (numMultiplicands != op.opCount().getZExtValue()) {
    return op.emitOpError("Number of multiplicands must match the specified operand count!");
  }
  return mlir::success();
}

static mlir::LogicalResult verify(SumOp op) {
  auto numAddends = std::distance(op.addends().begin(), op.addends().end());
  if (numAddends != op.opCount().getZExtValue()) {
    return op.emitOpError("Number of addends must match the specified operand count!");
  }
  return mlir::success();
}

static mlir::LogicalResult verify(WeightedSumOp op) {
  auto numAddends = std::distance(op.operands().begin(), op.operands().end());
  if (numAddends != op.opCount().getZExtValue()) {
    return op.emitOpError("Number of addends must match the specified operand count!");
  }
  auto numWeights = op.weights().size();
  if (numWeights != op.opCount().getZExtValue()) {
    return op.emitOpError("Number of weights must match the specified operand count!");
  }
  return mlir::success();
}

static mlir::LogicalResult verify(HistogramOp op) {
  int64_t lb = std::numeric_limits<int64_t>::min();
  int64_t ub = std::numeric_limits<int64_t>::min();
  auto buckets = op.buckets();
  for (auto b : buckets.getValue()) {
    auto bucket = b.cast<Bucket>();
    auto curLB = bucket.lb().getInt();
    auto curUB = bucket.ub().getInt();
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

static mlir::LogicalResult verify(SPNSingleQueryOp op) {
  auto* body = op.getBody();
  if (body->getNumArguments() != 1) {
    return op.emitOpError("Expected body to have a single argument for the input evidence!");
  }
  if (!body->getArgument(0).getType().isa<TensorType>()) {
    return op.emitOpError("Expected body argument to be a tensor!");
  }
  auto argType = body->getArgument(0).getType().cast<ShapedType>();
  auto evidenceType = op.input().getType().cast<ShapedType>();
  if (!argType.hasRank() || argType.getDimSize(0) != evidenceType.getDimSize(0)) {
    return op.emitOpError("Expected the body argument dimensions to match the input query!");
  }
  if (argType.getElementType() != evidenceType.getElementType()) {
    return op.emitOpError("Expected the body argument to have the same element type!");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/codegen/mlir/dialects/spn/SPNOps.cpp.inc"

