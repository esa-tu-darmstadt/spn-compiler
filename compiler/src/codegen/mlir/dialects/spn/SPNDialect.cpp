//
// Created by ls on 2/7/20.
//

#include "SPNDialect.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/DialectImplementation.h>
#include "mlir/IR/PatternMatch.h"
#include <codegen/mlir/transform/pattern/BinaryTreeTransformPatterns.h>

using namespace mlir;
using namespace mlir::spn;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
SPNDialect::SPNDialect(mlir::MLIRContext* ctx) : mlir::Dialect("spn", ctx) {
  addOperations<
#define GET_OP_LIST
#include "src/codegen/mlir/dialects/spn/SPNOps.cpp.inc"
  >();
  // We do not need to add the 'Bucket' attribute, because StructAttr are just
  // type-safe wrappers around DictionaryAttr.
}

//===----------------------------------------------------------------------===//
// ConstantOp

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::Builder *builder, mlir::OperationState &state,
                       double value) {
  ConstantOp::build(builder, state, builder->getF64Type(), builder->getF64FloatAttr(value));
}


static mlir::LogicalResult verify(ProductOp op) {
  auto numMultiplicands = std::distance(op.multiplicands().begin(), op.multiplicands().end());
  if (numMultiplicands != op.opCount().getZExtValue()) {
    return op.emitOpError("Number of multiplicands must match the specified operand count!");
  }
  return mlir::success();
}

void ProductOp::build(Builder* b, OperationState& state, llvm::ArrayRef<Value> operands) {
  build(b, state, b->getF64Type(), ValueRange(operands), b->getI32IntegerAttr(operands.size()));
}

void ProductOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<BinarizeProductOp>(context);
}

static mlir::LogicalResult verify(SumOp op) {
  auto numAddends = std::distance(op.addends().begin(), op.addends().end());
  if (numAddends != op.opCount().getZExtValue()) {
    return op.emitOpError("Number of addends must match the specified operand count!");
  }
  return mlir::success();
}

void SumOp::build(Builder* b, OperationState& state, llvm::ArrayRef<Value> operands) {
  build(b, state, b->getF64Type(), ValueRange(operands), b->getI32IntegerAttr(operands.size()));
}

void SumOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<BinarizeSumOp>(context);
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

void WeightedSumOp::build(Builder* b,
                          OperationState& state,
                          llvm::ArrayRef<Value> operands,
                          llvm::ArrayRef<double> weights) {
  SmallVector<mlir::Attribute, 10> weightAttrs;
  for (auto& w : weights) {
    weightAttrs.push_back(b->getF64FloatAttr(w));
  }
  assert(weightAttrs.size() == operands.size() && "Number of weights must match number of operands!");
  build(b, state, b->getF64Type(), ValueRange(operands), ArrayAttr::get(weightAttrs, b->getContext()),
        b->getI32IntegerAttr(operands.size()));
}

void WeightedSumOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<BinarizeWeightedSumOp>(context);
}

static mlir::LogicalResult verify(HistogramOp op) {
  int64_t lb = std::numeric_limits<int64_t>::min();
  int64_t ub = std::numeric_limits<int64_t>::min();
  if (op.buckets().size() != op.bucketCount().getZExtValue()) {
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

void HistogramOp::build(Builder* b, OperationState& state, Value index,
                        llvm::ArrayRef<std::tuple<int, int, double> > buckets) {
  SmallVector<mlir::Attribute, 256> bucketList;
  for (auto& bucket : buckets) {
    auto bucketAttr = Bucket::get(b->getI64IntegerAttr(std::get<0>(bucket)),
                                  b->getI64IntegerAttr(std::get<1>(bucket)),
                                  b->getF64FloatAttr(std::get<2>(bucket)), b->getContext());
    auto lb = b->getNamedAttr("lb", b->getI64IntegerAttr(std::get<0>(bucket)));
    auto ub = b->getNamedAttr("ub", b->getI64IntegerAttr(std::get<1>(bucket)));
    auto val = b->getNamedAttr("val", b->getF64FloatAttr(std::get<2>(bucket)));
    auto dictAttr = b->getDictionaryAttr({lb, ub, val});
    bucketList.push_back(dictAttr);
  }
  auto arrAttr = b->getArrayAttr(bucketList);
  build(b, state, b->getF64Type(), index, arrAttr, b->getI32IntegerAttr(bucketList.size()));
}

static mlir::LogicalResult verify(InputVarOp op) {
  if (!op.evidence().getType().isa<TensorType>()) {
    return op.emitOpError("Expected evidence argument to be a tensor!");
  }
  auto evidenceType = op.evidence().getType().cast<ShapedType>();
  // TODO Check if dimension 0 is correct here.
  if (!evidenceType.hasRank() || op.index().getZExtValue() >= evidenceType.getDimSize(0)) {
    return op.emitOpError("Index exceeds size of the evidence!");
  }
  return mlir::success();
}

void InputVarOp::build(Builder* b, OperationState& state, Value input, size_t index) {
  build(b, state, b->getIntegerType(32), input, b->getI32IntegerAttr((uint32_t) index));
}

CallInterfaceCallable SPNSingleQueryOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("spn");
}

Operation::operand_range SPNSingleQueryOp::getArgOperands() {
  return getODSOperands(0);
}

void SPNSingleQueryOp::build(Builder* b, OperationState& state, Value input, const std::string& callee) {
  build(b, state, b->getF64Type(), callee, input);
}

static mlir::LogicalResult verify(SPNSingleQueryOp op) {
  auto callee = op.getCallableForCallee();
  // TODO Verify that argument and return types match.
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/codegen/mlir/dialects/spn/SPNOps.cpp.inc"

