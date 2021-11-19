//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoGPU/GPUNodePatterns.h"
#include "LoSPN/LoSPNAttributes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <cmath>

mlir::LogicalResult mlir::spn::BatchReadGPULowering::matchAndRewrite(mlir::spn::low::SPNBatchRead op,
                                                                     llvm::ArrayRef<mlir::Value> operands,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  // Replace the BatchRead with a scalar load from the input memref,
  // using the batchIndex and the constant sample index.
  assert(operands.size() == 2 && "Expecting two operands for BatchRead");
  assert(operands[0].getType().isa<MemRefType>());
  auto memRefType = operands[0].getType().cast<MemRefType>();
  assert(memRefType.hasRank() && memRefType.getRank() == 2);
  assert(operands[1].getType().isa<IndexType>());
  SmallVector<Value> indices;
  auto constStaticIndex = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(op.staticIndex()));
  if (op.transposed().hasValue() && op.transposed().getValue()) {
    // Transposed access is memref[staticIndex][dynamicIndex]
    indices.push_back(constStaticIndex);
    indices.push_back(operands[1]);
  } else {
    // Non-transposed access is memref[dynamicIndex][staticIndex]
    indices.push_back(operands[1]);
    indices.push_back(constStaticIndex);
  }
  rewriter.replaceOpWithNewOp<memref::LoadOp>(op, operands[0], indices);
  return success();
}

mlir::LogicalResult mlir::spn::BatchWriteGPULowering::matchAndRewrite(mlir::spn::low::SPNBatchWrite op,
                                                                      llvm::ArrayRef<mlir::Value> operands,
                                                                      mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.size() == op.resultValues().size() + 2 && "Expecting correct number of operands for BatchWrite");
  // Replace the BatchWrite with stores to the input memref,
  // using the batchIndex.
  auto memRef = operands[0];
  auto memRefType = memRef.getType().dyn_cast<MemRefType>();
  assert(memRefType);
  assert(memRefType.hasRank() && memRefType.getRank() == 2);
  auto dynIndex = operands[1];
  assert(dynIndex.getType().isa<IndexType>());
  bool transposed = op.transposed().getValueOr(false);
  for (unsigned i = 0; i < op.resultValues().size(); ++i) {
    SmallVector<Value, 2> indices;
    auto constStaticIndex = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i));
    if (transposed) {
      indices.push_back(constStaticIndex);
      indices.push_back(dynIndex);
    } else {
      indices.push_back(dynIndex);
      indices.push_back(constStaticIndex);
    }
    rewriter.create<memref::StoreOp>(op.getLoc(), operands[i + 2], memRef, indices);
  }
  rewriter.eraseOp(op);
  return success();
}

mlir::LogicalResult mlir::spn::CopyGPULowering::matchAndRewrite(mlir::spn::low::SPNCopy op,
                                                                llvm::ArrayRef<mlir::Value> operands,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.size() == 2 && "Expecting two operands for Copy");
  assert(operands[0].getType().isa<MemRefType>());
  assert(operands[1].getType().isa<MemRefType>());
  assert(operands.size() == 2 && "Expecting two operands for Copy");
  assert(operands[0].getType().isa<MemRefType>());
  assert(operands[1].getType().isa<MemRefType>());
  auto srcType = op.source().getType().cast<MemRefType>();
  auto tgtType = op.target().getType().cast<MemRefType>();
  if (srcType.getRank() != tgtType.getRank() || srcType.getRank() != 1) {
    return rewriter.notifyMatchFailure(op, "Expecting one dimensional memories");
  }
  auto dim1 = rewriter.create<memref::DimOp>(op.getLoc(), op.source(), 0);
  auto lb = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  auto step = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1));
  auto outer = rewriter.create<scf::ForOp>(op.getLoc(), lb, dim1, step);
  rewriter.setInsertionPointToStart(&outer.getLoopBody().front());
  auto load = rewriter.create<memref::LoadOp>(op.getLoc(), op.source(), outer.getInductionVar());
  (void) rewriter.create<memref::StoreOp>(op.getLoc(), load, op.target(), outer.getInductionVar());
  rewriter.eraseOp(op);
  return success();
}

// Anonymous namespace holding helper functions.
mlir::LogicalResult mlir::spn::ConstantGPULowering::matchAndRewrite(mlir::spn::low::SPNConstant op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.empty() && "Expecting no operands for Constant");
  Type resultType = op.getResult().getType();
  if (auto logType = resultType.dyn_cast<low::LogType>()) {
    resultType = logType.getBaseType();
  }
  FloatAttr value = op.valueAttr();
  if (resultType != rewriter.getF64Type()) {
    assert(resultType.isa<FloatType>());
    value = rewriter.getFloatAttr(resultType, value.getValueAsDouble());
  }
  rewriter.replaceOpWithNewOp<ConstantOp>(op, resultType, value);
  return success();
}

mlir::LogicalResult mlir::spn::ReturnGPULowering::matchAndRewrite(mlir::spn::low::SPNReturn op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  if (!operands.empty()) {
    // At this point, all Tensor semantic should have been removed by the bufferization.
    // Hence, the SPNReturn, which can only return Tensors, should not have any return values anymore
    // and should merely be used as a terminator for Kernels and Tasks.
    return rewriter.notifyMatchFailure(op,
                                       "SPNReturn can only return Tensors, which should have been removed by bufferization");
  }
  rewriter.replaceOpWithNewOp<ReturnOp>(op);
  return success();
}

mlir::LogicalResult mlir::spn::LogGPULowering::matchAndRewrite(mlir::spn::low::SPNLog op,
                                                               llvm::ArrayRef<mlir::Value> operands,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.size() == 1 && "Expecting one operand for Log");
  rewriter.replaceOpWithNewOp<math::LogOp>(op, operands[0]);
  return success();
}

mlir::LogicalResult mlir::spn::MulGPULowering::matchAndRewrite(mlir::spn::low::SPNMul op,
                                                               llvm::ArrayRef<mlir::Value> operands,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  if (op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not match for log-space computation");
  }
  assert(operands.size() == 2 && "Expecting two operands for Mul");
  if (!operands[0].getType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Currently only matches floating-point multiplications");
  }
  rewriter.replaceOpWithNewOp<MulFOp>(op, operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::MulLogGPULowering::matchAndRewrite(mlir::spn::low::SPNMul op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  if (!op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches for log-space computation");
  }
  assert(operands.size() == 2 && "Expecting two operands for Mul");
  if (!operands[0].getType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Currently only matches floating-point multiplications");
  }
  rewriter.replaceOpWithNewOp<AddFOp>(op, operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::AddGPULowering::matchAndRewrite(mlir::spn::low::SPNAdd op,
                                                               llvm::ArrayRef<mlir::Value> operands,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  if (op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not match for log-space computation");
  }
  assert(operands.size() == 2 && "Expecting two operands for Add");
  if (!operands[0].getType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Currently only matches floating-point additions");
  }
  rewriter.replaceOpWithNewOp<AddFOp>(op, operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::AddLogGPULowering::matchAndRewrite(mlir::spn::low::SPNAdd op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  if (!op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches for log-space computation");
  }
  assert(operands.size() == 2 && "Expecting two operands for Mul");
  if (!operands[0].getType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Currently only matches floating-point multiplications");
  }
  // Calculate addition 'x + y' in log-space as
  // 'a + log(1 + exp(b-a)', with a == log(x),
  // b == log(y) and a > b.
  auto compare = rewriter.create<CmpFOp>(op.getLoc(), CmpFPredicate::OGT, operands[0], operands[1]);
  auto a = rewriter.create<SelectOp>(op->getLoc(), compare, operands[0], operands[1]);
  auto b = rewriter.create<SelectOp>(op->getLoc(), compare, operands[1], operands[0]);
  auto sub = rewriter.create<SubFOp>(op->getLoc(), b, a);
  auto exp = rewriter.create<math::ExpOp>(op.getLoc(), sub);
  // TODO Currently, GPULowering of log1p to LLVM is not supported,
  // therefore we perform the computation manually here.
  auto constantOne = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getFloatAttr(operands[0].getType(), 1.0));
  auto onePlus = rewriter.create<AddFOp>(op->getLoc(), constantOne, exp);
  auto log = rewriter.create<math::LogOp>(op.getLoc(), onePlus);
  rewriter.replaceOpWithNewOp<AddFOp>(op, a, log);
  return success();
}

mlir::LogicalResult mlir::spn::GaussianGPULowering::matchAndRewrite(mlir::spn::low::SPNGaussianLeaf op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  if (op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not match for log-space computation");
  }
  assert(operands.size() == 1 && "Expecting a single operands for Gaussian");
  if (!operands.front().getType().isIntOrFloat()) {
    return rewriter.notifyMatchFailure(op, "Pattern only takes int or float as input");
  }
  if (!op.getResult().getType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Cannot match Gaussian computing non-float result");
  }
  auto index = operands[0];
  auto resultType = op.getResult().getType().dyn_cast<FloatType>();

  auto indexType = index.getType();
  if (indexType.isIntOrIndex()) {
    // Convert integer/index input to floating point
    index = rewriter.create<UIToFPOp>(op->getLoc(), index, resultType);
  } else if (auto floatIndexType = indexType.dyn_cast<FloatType>()) {
    // Widden or narrow the index floating-point type to the result floating-point type.
    if (floatIndexType.getWidth() < resultType.getWidth()) {
      index = rewriter.create<mlir::FPExtOp>(op.getLoc(), index, resultType);
    } else if (floatIndexType.getWidth() > resultType.getWidth()) {
      index = rewriter.create<mlir::FPTruncOp>(op.getLoc(), index, resultType);
    }
  } else {
    // The input is neither float nor integer/index, fail this pattern because no conversion is possible.
    return rewriter.notifyMatchFailure(op, "Match failed because input is neither float nor integer/index");
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
  Value gaussian = rewriter.create<mlir::MulFOp>(op->getLoc(), coefficientConst, exp);
  if (op.supportMarginal()) {
    auto isNan = rewriter.create<mlir::CmpFOp>(op->getLoc(), CmpFPredicate::UNO, index, index);
    auto constOne = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getFloatAttr(resultType, 1.0));
    gaussian = rewriter.create<mlir::SelectOp>(op.getLoc(), isNan, constOne, gaussian);
  }
  rewriter.replaceOp(op, gaussian);
  return success();
}

mlir::LogicalResult mlir::spn::GaussianLogGPULowering::matchAndRewrite(mlir::spn::low::SPNGaussianLeaf op,
                                                                       llvm::ArrayRef<mlir::Value> operands,
                                                                       mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  if (!op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches for log-space computation");
  }
  assert(operands.size() == 1 && "Expecting a single operands for Gaussian");
  if (!operands.front().getType().isIntOrFloat()) {
    return rewriter.notifyMatchFailure(op, "Pattern only takes int or float as input");
  }
  if (!op.getResult().getType().cast<low::LogType>().getBaseType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Cannot match Gaussian computing non-float result");
  }
  auto index = operands[0];
  auto resultType = op.getResult().getType().cast<low::LogType>().getBaseType().dyn_cast<FloatType>();
  assert(resultType);

  auto indexType = index.getType();
  if (indexType.isIntOrIndex()) {
    // Convert integer/index input to floating point
    index = rewriter.create<UIToFPOp>(op->getLoc(), index, resultType);
  } else if (auto floatIndexType = indexType.dyn_cast<FloatType>()) {
    // Widden or narrow the index floating-point type to the result floating-point type.
    if (floatIndexType.getWidth() < resultType.getWidth()) {
      index = rewriter.create<mlir::FPExtOp>(op.getLoc(), index, resultType);
    } else if (floatIndexType.getWidth() > resultType.getWidth()) {
      index = rewriter.create<mlir::FPTruncOp>(op.getLoc(), index, resultType);
    }
  } else {
    // The input is neither float nor integer/index, fail this pattern because no conversion is possible.
    return rewriter.notifyMatchFailure(op, "Match failed because input is neither float nor integer/index");
  }

  // Calculate Gaussian distribution using the logarithm of the PDF of the Normal (Gaussian) distribution,
  // given as '-ln(stddev) - 1/2 ln(2*pi) - (x - mean)^2 / 2*stddev^2'
  // First term, -ln(stddev)
  double firstTerm = -log(op.stddev().convertToDouble());
  // Second term, - 1/2 ln(2*pi)
  double secondTerm = -0.5 * log(2 * M_PI);
  // Denominator, - 1/2*(stddev^2)
  double denominator = -(1.0 / (2.0 * op.stddev().convertToDouble() * op.stddev().convertToDouble()));
  auto denominatorConst = rewriter.create<mlir::ConstantOp>(op.getLoc(),
                                                            rewriter.getFloatAttr(resultType, denominator));
  // Coefficient, summing up the first two constant terms
  double coefficient = firstTerm + secondTerm;
  auto coefficientConst = rewriter.create<mlir::ConstantOp>(op->getLoc(),
                                                            rewriter.getFloatAttr(resultType, coefficient));
  // x - mean
  auto meanConst = rewriter.create<mlir::ConstantOp>(op.getLoc(),
                                                     rewriter.getFloatAttr(resultType,
                                                                           op.meanAttr().getValueAsDouble()));
  auto subtraction = rewriter.create<mlir::SubFOp>(op.getLoc(), index, meanConst);
  // (x-mean)^2
  auto numerator = rewriter.create<mlir::MulFOp>(op.getLoc(), subtraction, subtraction);
  // - ( (x-mean)^2 / 2 * stddev^2 )
  auto fraction = rewriter.create<mlir::MulFOp>(op.getLoc(), numerator, denominatorConst);
  // -ln(stddev) - 1/2 ln(2*pi) - 1/2*(stddev^2) * (x - mean)^2
  Value gaussian = rewriter.create<mlir::AddFOp>(op->getLoc(), coefficientConst, fraction);
  if (op.supportMarginal()) {
    auto isNan = rewriter.create<mlir::CmpFOp>(op->getLoc(), CmpFPredicate::UNO, index, index);
    auto constOne = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getFloatAttr(resultType, 0.0));
    gaussian = rewriter.create<mlir::SelectOp>(op.getLoc(), isNan, constOne, gaussian);
  }
  rewriter.replaceOp(op, gaussian);
  return success();
}

mlir::LogicalResult mlir::spn::CategoricalGPULowering::matchAndRewrite(mlir::spn::low::SPNCategoricalLeaf op,
                                                                       llvm::ArrayRef<mlir::Value> operands,
                                                                       mlir::ConversionPatternRewriter& rewriter) const {
  // Check for single operand, i.e., the index value.
  assert(operands.size() == 1);
  Type resultType = op.getResult().getType();
  bool computesLog = false;
  if (auto logType = resultType.dyn_cast<low::LogType>()) {
    resultType = logType.getBaseType();
    computesLog = true;
  }
  // Convert input value from float to integer if necessary.
  mlir::Value index = operands[0];
  if (!index.getType().isIntOrIndex()) {
    // If the input type is not an integer, but also not a float, we cannot convert it and this pattern fails.
    if (!index.getType().isIntOrFloat()) {
      return rewriter.notifyMatchFailure(op, "Cannot convert input of Categorical to integer");
    }
    index = rewriter.template create<mlir::FPToUIOp>(op.getLoc(), index, rewriter.getI64Type());
  }
  double defaultValue = (computesLog) ? static_cast<double>(-INFINITY) : 0;
  // TODO Replace 'getFloatAttr' with a more generic solution, if we want to support integer computation.
  Value falseVal = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getFloatAttr(resultType, defaultValue));
  auto probabilities = op.probabilitiesAttr().getValue();
  for (unsigned i = 0; i < op.probabilities().size(); ++i) {
    auto classVal = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getI64IntegerAttr(i));
    auto cmp = rewriter.create<CmpIOp>(op.getLoc(), CmpIPredicate::eq, index, classVal);
    auto probability = probabilities[i].dyn_cast<FloatAttr>().getValueAsDouble();
    if (computesLog) {
      probability = log(probability);
    }
    auto probVal = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getFloatAttr(resultType, probability));
    falseVal = rewriter.create<SelectOp>(op.getLoc(), cmp, probVal, falseVal);
  }
  auto indexOperand = operands[0];
  Value leaf = falseVal;
  if (op.supportMarginal()) {
    assert(indexOperand.getType().template isa<mlir::FloatType>());
    auto isNan = rewriter.create<mlir::CmpFOp>(op->getLoc(), mlir::CmpFPredicate::UNO,
                                               indexOperand, indexOperand);
    auto marginalValue = (computesLog) ? 0.0 : 1.0;
    auto constOne = rewriter.create<mlir::ConstantOp>(op.getLoc(),
                                                      rewriter.getFloatAttr(resultType, marginalValue));
    leaf = rewriter.create<mlir::SelectOp>(op.getLoc(), isNan, constOne, leaf);
  }
  rewriter.replaceOp(op, leaf);
  return mlir::success();
}

mlir::LogicalResult mlir::spn::HistogramGPULowering::matchAndRewrite(mlir::spn::low::SPNHistogramLeaf op,
                                                                     llvm::ArrayRef<mlir::Value> operands,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
  // Check for single operand, i.e., the index value.
  assert(operands.size() == 1);
  Type resultType = op.getResult().getType();
  bool computesLog = false;
  if (auto logType = resultType.dyn_cast<low::LogType>()) {
    resultType = logType.getBaseType();
    computesLog = true;
  }
  // Convert input value from float to integer if necessary.
  mlir::Value index = operands[0];
  if (!index.getType().isIntOrIndex()) {
    // If the input type is not an integer, but also not a float, we cannot convert it and this pattern fails.
    if (!index.getType().isIntOrFloat()) {
      return rewriter.notifyMatchFailure(op, "Cannot convert input of Categorical to integer");
    }
    index = rewriter.template create<mlir::FPToUIOp>(op.getLoc(), index, rewriter.getI64Type());
  }
  double defaultValue = (computesLog) ? static_cast<double>(-INFINITY) : 0;
  // TODO Replace 'getFloatAttr' with a more generic solution, if we want to support integer computation.
  Value falseVal = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getFloatAttr(resultType, defaultValue));
  SmallVector<low::Bucket> buckets;
  for (auto& b : op.bucketsAttr()) {
    auto bucket = b.cast<low::Bucket>();
    buckets.push_back(bucket);
  }

  auto indexOperand = operands[0];
  if (op.supportMarginal()) {
    assert(indexOperand.getType().template isa<mlir::FloatType>());
    auto isNan = rewriter.create<mlir::CmpFOp>(op->getLoc(), mlir::CmpFPredicate::UNO,
                                               indexOperand, indexOperand);
    auto marginalValue = (computesLog) ? 0.0 : 1.0;
    auto restore = rewriter.saveInsertionPoint();
    auto ifNaN = rewriter.create<scf::IfOp>(op.getLoc(), resultType, isNan, true);
    rewriter.setInsertionPointToStart(&ifNaN.thenRegion().front());
    Value constOne = rewriter.create<mlir::ConstantOp>(op.getLoc(),
                                                       rewriter.getFloatAttr(resultType, marginalValue));
    rewriter.create<scf::YieldOp>(op.getLoc(), constOne);
    rewriter.setInsertionPointToStart(&ifNaN.elseRegion().front());
    Value leaf = processBuckets(buckets, rewriter, index,
                                falseVal, op.getLoc(), computesLog);
    rewriter.create<scf::YieldOp>(op.getLoc(), leaf);
    rewriter.restoreInsertionPoint(restore);
    rewriter.replaceOp(op, ifNaN.getResult(0));
  } else {
    Value leaf = processBuckets(buckets, rewriter, index,
                                falseVal, op.getLoc(), computesLog);
    rewriter.replaceOp(op, leaf);
  }
  return mlir::success();
}

mlir::Value mlir::spn::HistogramGPULowering::processBuckets(llvm::ArrayRef<low::Bucket> buckets,
                                                            ConversionPatternRewriter& rewriter,
                                                            Value indexVal,
                                                            Value defaultVal,
                                                            Location loc,
                                                            bool computesLog) const {
  assert(!buckets.empty());
  auto restore = rewriter.saveInsertionPoint();
  if (buckets.size() == 1) {
    // Check that the index is actually in range of this bucket.
    auto lb = rewriter.create<ConstantOp>(loc,
                                          rewriter.getIntegerAttr(indexVal.getType(), buckets.front().lb().getInt()));
    auto ub = rewriter.create<ConstantOp>(loc,
                                          rewriter.getIntegerAttr(indexVal.getType(), buckets.front().ub().getInt()));
    auto lbCheck = rewriter.create<CmpIOp>(loc, CmpIPredicate::sge, indexVal, lb);
    auto ubCheck = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, indexVal, ub);
    auto rangeCheck = rewriter.create<AndOp>(loc, lbCheck, ubCheck);
    auto probability = buckets.front().val().getValueAsDouble();
    if (computesLog) {
      probability = std::log(probability);
    }
    auto probabilityVal = rewriter.create<ConstantOp>(loc, rewriter.getFloatAttr(defaultVal.getType(), probability));
    auto selectVal = rewriter.create<SelectOp>(loc, rangeCheck, probabilityVal, defaultVal);
    rewriter.restoreInsertionPoint(restore);
    return selectVal;
  } else {
    auto pivot = llvm::divideCeil(buckets.size(), 2);
    auto left = buckets.take_front(pivot);
    auto right = buckets.drop_front(pivot);
    auto border = rewriter.create<ConstantOp>(loc,
                                              rewriter.getIntegerAttr(indexVal.getType(), right.front().lb().getInt()));
    // Check if the index value is smaller then the first bucket of the right halve.
    auto compare = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, indexVal, border);
    auto ifOp = rewriter.create<scf::IfOp>(loc, defaultVal.getType(), compare, true);
    rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
    auto leftVal = processBuckets(left, rewriter, indexVal, defaultVal, loc, computesLog);
    rewriter.create<scf::YieldOp>(loc, leftVal);
    rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
    auto rightVal = processBuckets(right, rewriter, indexVal, defaultVal, loc, computesLog);
    rewriter.create<scf::YieldOp>(loc, rightVal);
    rewriter.restoreInsertionPoint(restore);
    return ifOp.getResult(0);
  }
}

mlir::LogicalResult mlir::spn::ResolveStripLogGPU::matchAndRewrite(mlir::spn::low::SPNStripLog op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not resolve vectorized operation");
  }
  assert(operands.size() == 1);
  if (operands[0].getType() != op.target()) {
    return rewriter.notifyMatchFailure(op, "Could not resolve StripLog trivially");
  }
  rewriter.replaceOp(op, operands[0]);
  return success();
}

mlir::LogicalResult mlir::spn::ResolveConvertLogGPU::matchAndRewrite(mlir::spn::low::SPNConvertLog op,
                                                                     llvm::ArrayRef<mlir::Value> operands,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.size() == 1);
  auto baseType = typeConverter->convertType(op.getResult().getType());
  assert(operands[0].getType() == baseType);
  // Simply replace the conversion by its input operand. All users of the conversion should be
  // converted subsequently.
  rewriter.replaceOp(op, operands[0]);
  return success();
}
