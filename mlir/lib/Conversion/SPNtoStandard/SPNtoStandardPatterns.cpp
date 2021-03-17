//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "SPNtoStandard/SPNtoStandardPatterns.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "SPN/SPNAttributes.h"

// Should not be necessary on modern platforms,
// but still defined for compatibility.
#define _USE_MATH_DEFINES
#include <math.h>

mlir::LogicalResult mlir::spn::ConstantOpLowering::matchAndRewrite(mlir::spn::ConstantOp op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, op.valueAttr());
  return success();
}

mlir::LogicalResult mlir::spn::ReturnOpLowering::matchAndRewrite(mlir::spn::ReturnOp op,
                                                                 llvm::ArrayRef<mlir::Value> operands,
                                                                 mlir::ConversionPatternRewriter& rewriter) const {
  return failure();
}

// Anonymous namespace holding helper functions.
namespace {

  template<typename SourceOp>
  mlir::LogicalResult replaceOpWithGlobalMemref(SourceOp op, mlir::ConversionPatternRewriter& rewriter,
                                                mlir::Value indexOperand, llvm::ArrayRef<mlir::Attribute> arrayValues) {
    static int tableCount = 0;
    auto resultType = op.getResult().getType();
    if (!resultType.isIntOrFloat()) {
      // Currently only handling Int and Float result types.
      return mlir::failure();
    }

    // Construct a DenseElementsAttr to hold the array values.
    auto rankedType = mlir::RankedTensorType::get({(long) arrayValues.size()}, resultType);
    auto valArrayAttr = mlir::DenseElementsAttr::get(rankedType, arrayValues);

    // Set the insertion point to the body of the module (outside the function/kernel).
    auto module = op->template getParentOfType<mlir::ModuleOp>();
    auto restore = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(module.getBody());

    // Construct a global, constant Memref with private visibility, holding the values of the array.
    auto symbolName = "table_" + std::to_string(tableCount++);
    auto visibility = rewriter.getStringAttr("private");
    auto memrefType = mlir::MemRefType::get({(long) arrayValues.size()}, resultType);
    auto globalMemref = rewriter.create<mlir::GlobalMemrefOp>(op.getLoc(), symbolName, visibility,
                                                              mlir::TypeAttr::get(memrefType), valArrayAttr, true);
    // Restore insertion point
    rewriter.restoreInsertionPoint(restore);

    // Use GetGlobalMemref operation to access the global created above.
    auto addressOf = rewriter.template create<mlir::GetGlobalMemrefOp>(op.getLoc(), memrefType, symbolName);
    // Convert input value from float to integer if necessary.
    mlir::Value index = indexOperand;
    if (!index.getType().isIntOrIndex()) {
      // If the input type is not an integer, but also not a float, we cannot convert it and this pattern fails.
      if (!index.getType().isIntOrFloat()) {
        return mlir::failure();
      }
      index = rewriter.template create<mlir::FPToUIOp>(op.getLoc(), index, rewriter.getI64Type());
    }
    // Cast input value to index if necessary.
    if (!index.getType().isIndex()) {
      index = rewriter.template create<mlir::IndexCastOp>(op.getLoc(), rewriter.getIndexType(), index);
    }
    // Replace the source operation with a load from the global memref,
    // using the source operation's input value as index.
    rewriter.template replaceOpWithNewOp<mlir::LoadOp>(op, addressOf, mlir::ValueRange{index});
    return mlir::success();
  }

}

mlir::LogicalResult mlir::spn::HistogramOpLowering::matchAndRewrite(mlir::spn::HistogramOp op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  // Check for single operand, i.e. the index value.
  assert(operands.size() == 1);

  // Collect all mappings from input var value to probability value in a map
  // and compute the minimum lower bound & maximum upper bound.
  llvm::DenseMap<int, double> values;
  int minLB = std::numeric_limits<int>::max();
  int maxUB = std::numeric_limits<int>::min();
  for (auto& b : op.bucketsAttr()) {
    auto bucket = b.cast<Bucket>();
    auto lb = bucket.lb().getInt();
    auto ub = bucket.ub().getInt();
    auto val = bucket.val().getValueAsDouble();
    for (int i = lb; i < ub; ++i) {
      values[i] = val;
    }
    minLB = std::min<int>(minLB, lb);
    maxUB = std::max<int>(maxUB, ub);
  }

  // Currently, we assume that all input vars take no values <0.
  if (minLB < 0) {
    return failure();
  }

  auto resultType = op.getResult().getType();
  if (!resultType.isIntOrFloat()) {
    // Currently only handling Int and Float result types.
    return failure();
  }

  // Flatten the map into an array by filling up empty indices with 0 values.
  SmallVector<Attribute, 256> valArray;
  for (int i = 0; i < maxUB; ++i) {
    double indexVal;
    if (values.count(i)) {
      indexVal = values[i];
    } else {
      // Fill up with 0 if no value was defined by the histogram.
      indexVal = 0;
    }
    // Construct attribute with constant value. Need to distinguish cases here due to different builder methods.
    if (resultType.isIntOrIndex()) {
      valArray.push_back(rewriter.getIntegerAttr(resultType, (int) indexVal));
    } else {
      valArray.push_back(rewriter.getFloatAttr(resultType, indexVal));
    }
  }

  return replaceOpWithGlobalMemref<HistogramOp>(op, rewriter, operands[0], valArray);
}

mlir::LogicalResult mlir::spn::CategoricalOpLowering::matchAndRewrite(mlir::spn::CategoricalOp op,
                                                                      llvm::ArrayRef<mlir::Value> operands,
                                                                      mlir::ConversionPatternRewriter& rewriter) const {
  // Check for single operand, i.e., the index value.
  assert(operands.size() == 1);

  return replaceOpWithGlobalMemref<CategoricalOp>(op, rewriter, operands[0],
                                                  op.probabilitiesAttr().getValue());
}

mlir::LogicalResult mlir::spn::GaussianOpLowering::matchAndRewrite(mlir::spn::GaussianOp op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.size() == 1);
  Value index = operands[0];
  if (!op.getResult().getType().isa<FloatType>() || index.getType().isa<VectorType>()) {
    // Can only compute scalar floating-point results.
    return failure();
  }
  auto resultType = op.getResult().getType().dyn_cast<FloatType>();

  Type indexType = index.getType();
  if (indexType.isIntOrIndex()) {
    // Convert to floating point
    index = rewriter.create<UIToFPOp>(op.getLoc(), index, resultType);
  } else if (indexType.isa<FloatType>()) {
    auto fInType = indexType.dyn_cast<FloatType>();
    // Widen or narrow the index floating-point type to the result floating-point type.
    if (fInType.getWidth() < resultType.getWidth()) {
      index = rewriter.create<mlir::FPExtOp>(op.getLoc(), index, resultType);
    } else if (fInType.getWidth() > resultType.getWidth()) {
      index = rewriter.create<mlir::FPTruncOp>(op.getLoc(), index, resultType);
    }
  } else {
    // The input type is neither float nor integer, fail this pattern because no conversion possible.
    return failure();
  }
  // Calculate Gaussian distribution using e^(-(x - mean)^2/2*variance))/sqrt(2*PI*variance)
  // Variance from standard deviation.
  double variance = op.stddev().convertToDouble() * op.stddev().convertToDouble();
  // 1/sqrt(2*PI*variance)
  double coefficient = 1.0 / (std::sqrt(2.0 * M_PI * variance));
  auto coefficientConst = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getF64FloatAttr(coefficient));
  // -1/(2*variance)
  double denominator = -1.0 / (2.0 * variance);
  auto denominatorConst = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getF64FloatAttr(denominator));
  // x - mean
  auto meanConst = rewriter.create<mlir::ConstantOp>(op.getLoc(), op.meanAttr());
  auto subtraction = rewriter.create<mlir::SubFOp>(op.getLoc(), index, meanConst);
  // (x-mean)^2
  auto numerator = rewriter.create<mlir::MulFOp>(op.getLoc(), subtraction, subtraction);
  // -(x-mean)^2 / 2*variance
  auto fraction = rewriter.create<mlir::MulFOp>(op.getLoc(), numerator, denominatorConst);
  // e^(-(x-mean)^2 / 2*variance)
  auto exp = rewriter.create<mlir::math::ExpOp>(op.getLoc(), fraction);
  // e^(-(x - mean)^2/2*variance)) * 1/sqrt(2*PI*variance)
  rewriter.replaceOpWithNewOp<mlir::MulFOp>(op, coefficientConst, exp);
  return success();
}

mlir::LogicalResult mlir::spn::SingleJointLowering::matchAndRewrite(mlir::spn::JointQuery op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  // This lowering is specialized for single evaluations, reject queries with batch size >1.
  if (dyn_cast<QueryInterface>(op.getOperation()).getBatchSize() > 1) {
    return failure();
  }

  auto inputType = MemRefType::get({op.numFeatures()}, op.inputType());
  auto returnOp = op.graph().front().getTerminator();
  auto graphResult = dyn_cast<mlir::spn::ReturnOp>(returnOp);
  assert(graphResult);
  auto resultType = MemRefType::get({1}, graphResult.retValue().front().getType());

  auto replaceFunc = rewriter.create<FuncOp>(op.getLoc(), op.kernelName(),
                                             rewriter.getFunctionType({inputType, resultType}, llvm::None),
                                             llvm::None);

  auto funcEntryBlock = replaceFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(funcEntryBlock);
  auto inputArg = replaceFunc.getArgument(0);
  assert(inputArg.getType().isa<MemRefType>());
  SmallVector<Value, 10> blockArgsReplacement;
  for (size_t i = 0; i < op.numFeatures(); ++i) {
    SmallVector<Value, 1> indices;
    indices.push_back(rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i)));
    auto load = rewriter.create<mlir::LoadOp>(op.getLoc(), inputArg, indices);
    blockArgsReplacement.push_back(load);
  }
  rewriter.mergeBlocks(&op.getRegion().front(), funcEntryBlock, blockArgsReplacement);
  rewriter.setInsertionPointToEnd(funcEntryBlock);
  // Apply logarithm to result before storing it
  auto logResult = rewriter.create<mlir::math::LogOp>(op.getLoc(), graphResult.retValue().front());
  // Store the log-result to the output pointer.
  SmallVector<Value, 1> indices;
  indices.push_back(rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0)));
  rewriter.create<mlir::StoreOp>(op.getLoc(), logResult,
                                 replaceFunc.getArgument(1), indices);
  rewriter.create<mlir::ReturnOp>(op.getLoc());
  rewriter.eraseOp(graphResult);
  rewriter.eraseOp(op);
  return success();
}

mlir::LogicalResult mlir::spn::BatchJointLowering::matchAndRewrite(mlir::spn::JointQuery op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  // This lowering is specialized for batch evaluation, reject queries with batch size <= 1.
  unsigned batchSize = dyn_cast<QueryInterface>(op.getOperation()).getBatchSize();
  if (batchSize <= 1) {
    return failure();
  }

  // Create function with three inputs:
  //   * Number of elements to process (might be lower than the batch size).
  //   * Pointer to the input data.
  //   * Pointer to the output data.
  auto numSamplesType = rewriter.getI64Type();
  auto inputType = MemRefType::get({batchSize, op.numFeatures()}, op.inputType());
  auto returnOp = op.graph().front().getTerminator();
  auto graphResult = dyn_cast<mlir::spn::ReturnOp>(returnOp);
  assert(graphResult);
  auto resultType = MemRefType::get({batchSize}, graphResult.retValue().front().getType());

  auto replaceFunc = rewriter.create<FuncOp>(op.getLoc(), op.kernelName(),
                                             rewriter.getFunctionType(
                                                 {numSamplesType, inputType, resultType}, llvm::None),
                                             llvm::None);

  auto funcEntryBlock = replaceFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(funcEntryBlock);
  auto numSamplesArg = replaceFunc.getArgument(0);
  auto inputArg = replaceFunc.getArgument(1);
  assert(inputArg.getType().isa<MemRefType>());

  // Construct a for-loop iterating over the samples. Upper bound on the number of iterations is the
  // numSamples argument to the function, assuming that this argument is set to batchSize except when
  // processing the remainders of the data-set (i.e. less than batchSize samples).
  auto lb = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  auto ub = rewriter.create<mlir::IndexCastOp>(op.getLoc(), numSamplesArg, rewriter.getIndexType());
  auto step = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1));
  auto forLoop = rewriter.create<mlir::scf::ForOp>(op.getLoc(), lb, ub, step);
  auto& loopBody = forLoop.getLoopBody().front();

  // Load input values in the loop body
  rewriter.setInsertionPointToStart(&loopBody);
  SmallVector<Value, 10> blockArgsReplacement;
  for (size_t i = 0; i < op.numFeatures(); ++i) {
    SmallVector<Value, 2> indices;
    indices.push_back(forLoop.getInductionVar());
    indices.push_back(rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i)));
    auto load = rewriter.create<mlir::LoadOp>(op.getLoc(), inputArg, indices);
    blockArgsReplacement.push_back(load);
  }
  // Transfer SPN graph from JointQuery to for-loop body.
  rewriter.mergeBlockBefore(&op.getRegion().front(), loopBody.getTerminator(), blockArgsReplacement);
  // Apply logarithm to result before storing it
  rewriter.setInsertionPoint(loopBody.getTerminator());
  auto logResult = rewriter.create<mlir::math::LogOp>(op.getLoc(), graphResult.retValue().front());
  // Store the log-result to the output pointer.
  SmallVector<Value, 1> indices;
  indices.push_back(forLoop.getInductionVar());
  rewriter.create<mlir::StoreOp>(op.getLoc(), logResult,
                                 replaceFunc.getArgument(2), indices);
  // Insert a return as terminator into the function's block.
  rewriter.setInsertionPointToEnd(funcEntryBlock);
  rewriter.create<mlir::ReturnOp>(op.getLoc());
  // Remove the old return from the SPN dialect and the query operation (JointQuery) itself.
  rewriter.eraseOp(graphResult);
  rewriter.eraseOp(op);
  return success();
}
