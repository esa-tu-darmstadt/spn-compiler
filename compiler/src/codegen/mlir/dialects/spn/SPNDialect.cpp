//
// Created by ls on 2/7/20.
//

#include "SPNDialect.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/DialectImplementation.h>
#include "mlir/IR/PatternMatch.h"
#include <codegen/mlir/transform/pattern/CanonicalizationPatterns.h>

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
void ConstantOp::build(mlir::Builder* builder, mlir::OperationState& state,
                       double value) {
  ConstantOp::build(builder, state, builder->getF64Type(), builder->getF64FloatAttr(value));
}

void ReturnOp::build(Builder* b, OperationState& state) {
  build(b, state, llvm::None);
}

void ReturnOp::build(Builder* b, OperationState& state, Value retValue) {
  build(b, state, ValueRange{retValue});
}

template<typename NAryOp>
static mlir::LogicalResult verify(NAryOp op) {
  auto numOperands = std::distance(op.operands().begin(), op.operands().end());
  if (numOperands != op.opCount().getZExtValue()) {
    return op.emitOpError("Number of operands must match the specified operand count!");
  }
  return mlir::success();
}

template mlir::LogicalResult verify<ProductOp>(ProductOp op);

void ProductOp::build(Builder* b, OperationState& state, llvm::ArrayRef<Value> operands) {
  build(b, state, b->getF64Type(), ValueRange(operands), b->getI32IntegerAttr(operands.size()));
}

void ProductOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<ReduceProductOp>(context);
  results.insert<ConstantFoldProductOp>(context);
}

template mlir::LogicalResult verify<SumOp>(SumOp op);

void SumOp::build(Builder* b, OperationState& state, llvm::ArrayRef<Value> operands) {
  build(b, state, b->getF64Type(), ValueRange(operands), b->getI32IntegerAttr(operands.size()));
}

void SumOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<ReduceSumOp>(context);
  results.insert<ConstantFoldSumOp>(context);
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
  results.insert<ReduceWeightedSumOp>(context);
  results.insert<ConstantFoldWeightedSumOp>(context);
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

static mlir::LogicalResult verify(HistogramValueOp op) {
  if (op.ub().getZExtValue() <= op.lb().getZExtValue()) {
    return op.emitOpError("Upper bound must be strictly greater than lower bound");
  }
  if (op.values().getNumElements() != (op.ub() - op.lb()).getZExtValue()) {
    return op.emitOpError("Array count must match lower- and upper bound");
  }
  return mlir::success();
}

void HistogramValueOp::build(Builder* b, OperationState& state, llvm::ArrayRef<double> values, int lb, int ub) {
  auto lbAttr = b->getI32IntegerAttr(lb);
  auto ubAttr = b->getI32IntegerAttr(ub);
  SmallVector<mlir::Attribute, 256> valueAttributes;
  for (auto d : values) {
    valueAttributes.push_back(b->getF64FloatAttr(d));
  }
  auto rankedType = RankedTensorType::get({(long) values.size()}, b->getF64Type());
  auto valuesAttr = DenseFPElementsAttr::get(rankedType, values);
  auto memRefType = MemRefType::get({(ub - lb)}, b->getF64Type());
  build(b, state, memRefType, valuesAttr, lbAttr, ubAttr);
}

static mlir::LogicalResult verify(InputVarOp op) {
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

static mlir::LogicalResult verifyQuery(QueryInterface op) {
  if (auto callOp = dyn_cast<CallOpInterface>(&*op)) {
    auto callee = callOp.resolveCallable();
    if (auto funcOp = dyn_cast<FuncOp>(callee)) {
      // Check for correct argument count and type.
      if (funcOp.getNumArguments() != 1 || !funcOp.getArgument(0).getType().isa<TensorType>()) {
        op.emitOpError("Callee should only have a single tensor as parameter!");
      }
      // Check for correct shape and element type of the single input argument.
      auto arg1 = funcOp.getArgument(0).getType().cast<ShapedType>();
      auto inputShape = op.getNumFeatures();
      if (!arg1.hasRank() || !arg1.hasStaticShape() || arg1.getDimSize(0) != inputShape
          || arg1.getElementType() != op.getFeatureType()) {
        op.emitOpError("Callee parameter must have same static size and type as input operand!");
      }
      // Check for correct count and type of the callee return value.
      if (funcOp.getType().getNumResults() != 1 || funcOp.getType().getResult(0) != op.getResultType()) {
        op.emitOpError("Callee must return a single value of matching type!");
      }
    }
  }
  return mlir::success();
}

CallInterfaceCallable SPNSingleQueryOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("spn");
}

Operation::operand_range SPNSingleQueryOp::getArgOperands() {
  return getODSOperands(0);
}

int SPNSingleQueryOp::getNumFeatures() {
  return input().getType().cast<ShapedType>().getDimSize(0);
}

Type SPNSingleQueryOp::getFeatureType() {
  return input().getType().cast<ShapedType>().getElementType();
}

Type SPNSingleQueryOp::getResultType() {
  return this->getType();
}

void SPNSingleQueryOp::build(Builder* b, OperationState& state, Value input, const std::string& callee) {
  build(b, state, b->getF64Type(), callee, input);
}

static mlir::LogicalResult verify(SPNSingleQueryOp op) {
  auto inputType = op.input().getType().cast<ShapedType>();

  if (!inputType.hasRank() || inputType.getRank() != 1 || ShapedType::isDynamic(inputType.getDimSize(0))) {
    op.emitOpError("Expected input to be a 1-dimensional tensor with static dimension!");
  }

  if (auto queryOp = dyn_cast<QueryInterface>(&*op)) {
    verifyQuery(queryOp);
  }
  return mlir::success();
}

CallInterfaceCallable SPNJointProbBatch::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("spn");
}

Operation::operand_range SPNJointProbBatch::getArgOperands() {
  return getODSOperands(0);
}

int SPNJointProbBatch::getNumFeatures() {
  return input().getType().cast<ShapedType>().getDimSize(1);
}

Type SPNJointProbBatch::getFeatureType() {
  return input().getType().cast<ShapedType>().getElementType();
}

Type SPNJointProbBatch::getResultType() {
  return output().getType().cast<ShapedType>().getElementType();
}

void SPNJointProbBatch::build(Builder* b, OperationState& state, Value input, Value output, const std::string& callee) {
  build(b, state, callee, input, output);
}

static mlir::LogicalResult verify(SPNJointProbBatch op) {
  auto inputType = op.input().getType().cast<ShapedType>();
  auto outputType = op.output().getType().cast<ShapedType>();

  if (!inputType.hasRank() || inputType.getRank() != 2 || ShapedType::isDynamic(inputType.getDimSize(1))) {
    op.emitOpError("Expected input to be a 2-dimensional tensor with static second dimension!");
  }

  if (!outputType.hasRank() || outputType.getRank() != 1) {
    op.emitOpError("Expected output to be a 1-dimensional tensor!");
  }

  if (auto queryOp = dyn_cast<QueryInterface>(&*op)) {
    verifyQuery(queryOp);
  }
  return mlir::success();

}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/codegen/mlir/dialects/spn/SPNOps.cpp.inc"

