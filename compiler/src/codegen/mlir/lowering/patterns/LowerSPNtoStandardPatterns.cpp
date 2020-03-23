//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LowerSPNtoStandardPatterns.h"
#include "SPNOperationLowering.h"
#include <mlir/IR/Matchers.h>
#include <limits>
#include <util/Logging.h>

using namespace mlir;
using namespace mlir::spn;

PatternMatchResult ConstantOpLowering::matchAndRewrite(spn::ConstantOp op, PatternRewriter& rewriter) const {
  // Simply replace the operation with the equivalent from the Standard dialect.
  rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, op.valueAttr());
  return matchSuccess();
}

PatternMatchResult ReturnOpLowering::matchAndRewrite(spn::ReturnOp op, PatternRewriter& rewriter) const {
  // Simply replace the operation with the equivalent from the Standard dialect.
  rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, op.retValue());
  return matchSuccess();
}

PatternMatchResult InputVarLowering::matchAndRewrite(InputVarOp op, ArrayRef<Value> operands,
                                                     ConversionPatternRewriter& rewriter) const {
  // InputVarOps are now only used to associate some high-level information
  // about the input variable with the input to the SPN-function's argument
  // at the corresponding index. We can simply replace the InputVarOp with
  // its single argument.
  rewriter.replaceOp(op, {operands[0]});
  return matchSuccess();
}

PatternMatchResult FunctionLowering::matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                                                     ConversionPatternRewriter& rewriter) const {
  // As part of this lowering, we convert all Tensor-values to MemRef-values.
  // Therefore, we need to rewrite the signature of the functions to convert
  // the types of the function arguments.
  auto fnType = op.getType();

  // Conversion is currently limited to functions with a single argument.
  if (fnType.getNumResults() > 1) {
    SPDLOG_ERROR("Function returns more than one result!");
    return matchFailure();
  }

  // Get the result type or None, if the function does not return a value.
  SmallVector<Type, 1> results;
  for (auto& r : fnType.getResults()) {
    auto convertedType = typeConverter.convertType(r);
    if (!convertedType) {
      SPDLOG_ERROR("Could not convert function return type");
      return matchFailure();
    }
    results.push_back(convertedType);
  }
  //auto resType = (fnType.getNumResults()) ? (ArrayRef<Type>) {fnType.getResults()[0]} : llvm::None;

  // Use a SignatureConversion to convert all argument types.
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  for (auto argType : llvm::enumerate(fnType.getInputs())) {
    if (failed(typeConverter.convertSignatureArg((unsigned) argType.index(), argType.value(), signatureConverter))) {
      SPDLOG_ERROR("Could not convert function argument type");
      return matchFailure();
    }
  }

  // Create a new function with the same name, but converted signature.
  auto newFuncOp = rewriter.create<FuncOp>(op.getLoc(), op.getName(),
                                           rewriter.getFunctionType(signatureConverter.getConvertedTypes(), results),
                                           llvm::None);

  // Copy over all attributes, except for name- and type-attributes.
  for (const auto& namedAttr : op.getAttrs()) {
    if (!namedAttr.first.is(impl::getTypeAttrName()) &&
        !namedAttr.first.is(SymbolTable::getSymbolAttrName())) {
      newFuncOp.setAttr(namedAttr.first, namedAttr.second);
    }
  }

  // Move the body of the function to the new function.
  rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(), newFuncOp.end());
  // Inform all operations in the body about the conversion of the function arguments.
  rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);
  // Delete the old, now unused function to avoid name conflicts.
  rewriter.eraseOp(op);
  return matchSuccess();
}

PatternMatchResult HistogramLowering::matchAndRewrite(HistogramOp op, ArrayRef<Value> operands,
                                                      ConversionPatternRewriter& rewriter) const {
  // Lower histogram. As the Standard dialect currently (2020/02/28) does not support Tensor-
  // or vector-constants, we will insert a HistogramValueOp, which contains a flattened array
  // of probability values from the histogram and will be converted to an actual array when
  // converting to LLVM dialect (or similar) later on.

  // Collect all mappings from input var value to probability value in a map.
  llvm::DenseMap<int, double> values;
  int minLB = std::numeric_limits<int>::max();
  int maxUB = std::numeric_limits<int>::min();
  for (auto& b : op.bucketsAttr()) {
    auto dict = b.cast<DictionaryAttr>();
    auto lb = dict.get("lb").cast<IntegerAttr>().getInt();
    auto ub = dict.get("ub").cast<IntegerAttr>().getInt();
    auto val = dict.get("val").cast<FloatAttr>().getValueAsDouble();
    for (int i = lb; i < ub; ++i) {
      values[i] = val;
    }
    minLB = std::min<int>(minLB, lb);
    maxUB = std::max<int>(maxUB, ub);
  }

  // Currently, we assume that all input vars take no values <0.
  if (minLB < 0) {
    return matchFailure();
  }

  // Flatten the map into an array by filling up empty indices with 0 values.
  SmallVector<double, 256> valArray;
  for (int i = 0; i < maxUB; ++i) {
    if (values.count(i)) {
      valArray.push_back(values[i]);
    } else {
      valArray.push_back(0.0);
    }
  }

  // Insert a HistogramValueOp, which return a MemRef to the array of constant values.
  auto histValueAray = rewriter.create<HistogramValueOp>(op.getLoc(), valArray, 0, maxUB);
  // We need to cast the input var value to index type, in order to be able to use it for loads.
  auto castIndex = rewriter.create<mlir::IndexCastOp>(op.getLoc(), op.index(), rewriter.getIndexType());
  // Replace the histogram with a load from the array of constant values.
  auto histRead = rewriter.create<mlir::LoadOp>(op.getLoc(), histValueAray, ValueRange{castIndex});
  rewriter.replaceOp(op, {histRead});
  return matchSuccess();
}

PatternMatchResult SingleQueryLowering::matchAndRewrite(SPNSingleQueryOp op, ArrayRef<Value> operands,
                                                        ConversionPatternRewriter& rewriter) const {
  SmallVector<Type, 1> retType{rewriter.getF64Type()};
  if (!operands[0].getType().isa<MemRefType>()) {
    return matchFailure();
  }
  auto memRefType = operands[0].getType().cast<MemRefType>();
  if (!memRefType.hasStaticShape() || MemRefType::isDynamic(memRefType.getDimSize(0))) {
    return matchFailure();
  }
  // Load each feature value from the memref (was a tensor before lowering) and
  // replace SPNSingleQuery by a call to the SPN function.
  auto memRef = operands[0];
  auto numFeatures = memRefType.getDimSize(0);
  SmallVector<Value, 10> loads;
  for (size_t i = 0; i < numFeatures; ++i) {
    auto load = createStaticLoad(rewriter, op.getLoc(), memRef, {i});
    loads.push_back(load);
  }
  rewriter.replaceOpWithNewOp<mlir::CallOp>(op, retType, op.spnAttr(), loads);
  return matchSuccess();
}

template<typename SourceOp>
LoadOp SPNOpLowering<SourceOp>::createStaticLoad(mlir::ConversionPatternRewriter& rewriter,
                                                 mlir::Location loc,
                                                 mlir::Value memRef,
                                                 llvm::ArrayRef<size_t> indices) const {
  assert(memRef.getType().isa<MemRefType>() && "Base address must be a memref!");
  assert(!indices.empty() && "Expecting at least one static index!");
  SmallVector<Value, 10> indexValues;
  for (auto i : indices) {
    auto constantIndex = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getIndexType(), i));
    indexValues.push_back(constantIndex);
  }
  return rewriter.create<mlir::LoadOp>(loc, memRef, indexValues);
}