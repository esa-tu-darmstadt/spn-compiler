//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/NodePatterns.h"
#include <cmath>
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "LoSPN/LoSPNAttributes.h"

mlir::LogicalResult mlir::spn::BatchReadLowering::matchAndRewrite(mlir::spn::low::SPNBatchRead op,
                                                                  OpAdaptor adaptor,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
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
  auto constStaticIndex = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(op.getStaticIndex()));
  if (op.getTransposed().value_or(false)) {
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

mlir::LogicalResult mlir::spn::BatchWriteLowering::matchAndRewrite(mlir::spn::low::SPNBatchWrite op,
                                                                   OpAdaptor adaptor,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.size() == op.getResultValues().size() + 2 && "Expecting correct number of operands for BatchWrite");
  // Replace the BatchWrite with stores to the input memref,
  // using the batchIndex.
  auto memRef = operands[0];
  auto memRefType = memRef.getType().dyn_cast<MemRefType>();
  assert(memRefType);
  assert(memRefType.hasRank() && memRefType.getRank() == 2);
  auto dynIndex = operands[1];
  assert(dynIndex.getType().isa<IndexType>());
  bool transposed = op.getTransposed().value_or(false);
  for (unsigned i = 0; i < op.getResultValues().size(); ++i) {
    SmallVector<Value, 2> indices;
    auto constStaticIndex = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i));
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

mlir::LogicalResult mlir::spn::CopyLowering::matchAndRewrite(mlir::spn::low::SPNCopy op,
                                                             OpAdaptor adaptor,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
  assert(operands.size() == 2 && "Expecting two operands for Copy");
  assert(operands[0].getType().isa<MemRefType>());
  assert(operands[1].getType().isa<MemRefType>());
  auto srcType = op.getSource().getType().cast<MemRefType>();
  auto tgtType = op.getTarget().getType().cast<MemRefType>();
  assert(srcType.hasRank());
  assert(tgtType.hasRank());
  if (srcType.getRank() != tgtType.getRank() || srcType.getRank() != 2) {
    return rewriter.notifyMatchFailure(op, "Expecting two dimensional memories");
  }
  if (srcType.getShape() != tgtType.getShape()) {
    return rewriter.notifyMatchFailure(op, "Shape of both arguments must match");
  }
  assert(srcType.isDynamicDim(0) ^ srcType.isDynamicDim(1));
  auto transposed = srcType.isDynamicDim(1);
  auto dynIdx = (!transposed) ? 0 : 1;
  auto staticIdx = (!transposed) ? 1 : 0;
  auto staticDim = srcType.getDimSize(staticIdx);
  assert(staticDim > 0);
  auto dim1 = rewriter.create<memref::DimOp>(op.getLoc(), op.getSource(), dynIdx);
  auto lb = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  auto step = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1));
  auto outer = rewriter.create<scf::ForOp>(op.getLoc(), lb, dim1, step);
  rewriter.setInsertionPointToStart(&outer.getLoopBody().front());
  for (int i = 0; i < staticDim; ++i) {
    SmallVector<Value, 2> indices;
    auto constIdx = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i));
    if (transposed) {
      indices.push_back(constIdx);
      indices.push_back(outer.getInductionVar());
    } else {
      indices.push_back(outer.getInductionVar());
      indices.push_back(constIdx);
    }
    auto load = rewriter.create<memref::LoadOp>(op.getLoc(), op.getSource(), indices);
    (void) rewriter.create<memref::StoreOp>(op.getLoc(), load, op.getTarget(), indices);
  }
  rewriter.eraseOp(op);
  return success();
}

// Anonymous namespace holding helper functions.
mlir::LogicalResult mlir::spn::ConstantLowering::matchAndRewrite(mlir::spn::low::SPNConstant op,
                                                                 OpAdaptor adaptor,
                                                                 mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.empty() && "Expecting no operands for Constant");
  Type resultType = op.getResult().getType();
  if (auto logType = resultType.dyn_cast<low::LogType>()) {
    resultType = logType.getBaseType();
  }
  FloatAttr value = op.getValueAttr();
  if (resultType != rewriter.getF64Type()) {
    assert(resultType.isa<FloatType>());
    value = rewriter.getFloatAttr(resultType, value.getValueAsDouble());
  }
  rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, value);
  return success();
}

mlir::LogicalResult mlir::spn::ReturnLowering::matchAndRewrite(mlir::spn::low::SPNReturn op,
                                                               OpAdaptor adaptor,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
  if (!operands.empty()) {
    // At this point, all Tensor semantic should have been removed by the bufferization.
    // Hence, the SPNReturn, which can only return Tensors, should not have any return values anymore
    // and should merely be used as a terminator for Kernels and Tasks.
    return rewriter.notifyMatchFailure(op,
                                       "SPNReturn can only return Tensors, which should have been removed by bufferization");
  }
  rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op);
  return success();
}

mlir::LogicalResult mlir::spn::LogLowering::matchAndRewrite(mlir::spn::low::SPNLog op,
                                                            OpAdaptor adaptor,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.size() == 1 && "Expecting one operand for Log");
  rewriter.replaceOpWithNewOp<math::LogOp>(op, operands[0]);
  return success();
}

mlir::LogicalResult mlir::spn::MulLowering::matchAndRewrite(mlir::spn::low::SPNMul op,
                                                            OpAdaptor adaptor,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
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
  rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(op, operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::MulLogLowering::matchAndRewrite(mlir::spn::low::SPNMul op,
                                                               OpAdaptor adaptor,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
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
  rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op, operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::AddLowering::matchAndRewrite(mlir::spn::low::SPNAdd op,
                                                            OpAdaptor adaptor,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
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
  rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op, operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::AddLogLowering::matchAndRewrite(mlir::spn::low::SPNAdd op,
                                                               OpAdaptor adaptor,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
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
  auto compare = rewriter.create<mlir::arith::CmpFOp>(op.getLoc(), arith::CmpFPredicate::OGT, operands[0], operands[1]);
  auto a = rewriter.create<mlir::arith::SelectOp>(op->getLoc(), compare, operands[0], operands[1]);
  auto b = rewriter.create<mlir::arith::SelectOp>(op->getLoc(), compare, operands[1], operands[0]);
  auto sub = rewriter.create<mlir::arith::SubFOp>(op->getLoc(), b, a);
  auto exp = rewriter.create<mlir::math::ExpOp>(op.getLoc(), sub);
  auto log = rewriter.create<mlir::math::Log1pOp>(op.getLoc(), exp);
  rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op, a, log);
  return success();
}

mlir::LogicalResult mlir::spn::GaussianLowering::matchAndRewrite(mlir::spn::low::SPNGaussianLeaf op,
                                                                 OpAdaptor adaptor,
                                                                 mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
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
    index = rewriter.create<mlir::arith::UIToFPOp>(op->getLoc(), resultType, index);
  } else if (auto floatIndexType = indexType.dyn_cast<FloatType>()) {
    // Widden or narrow the index floating-point type to the result floating-point type.
    if (floatIndexType.getWidth() < resultType.getWidth()) {
      index = rewriter.create<mlir::arith::ExtFOp>(op.getLoc(), resultType, index);
    } else if (floatIndexType.getWidth() > resultType.getWidth()) {
      index = rewriter.create<mlir::arith::TruncIOp>(op.getLoc(), resultType, index);
    }
  } else {
    // The input is neither float nor integer/index, fail this pattern because no conversion is possible.
    return rewriter.notifyMatchFailure(op, "Match failed because input is neither float nor integer/index");
  }

  // Calculate Gaussian distribution using e^(-(x - mean)^2/2*variance))/sqrt(2*PI*variance)
  // Variance from standard deviation.
  double variance = op.getStddev().convertToDouble() * op.getStddev().convertToDouble();
  // 1/sqrt(2*PI*variance)
  double coefficient = 1.0 / (std::sqrt(2.0 * M_PI * variance));
  auto coefficientConst = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), rewriter.getF64FloatAttr(coefficient));
  // -1/(2*variance)
  double denominator = -1.0 / (2.0 * variance);
  auto denominatorConst = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), rewriter.getF64FloatAttr(denominator));
  // x - mean
  auto meanConst = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), op.getMeanAttr());
  auto subtraction = rewriter.create<mlir::arith::SubFOp>(op.getLoc(), index, meanConst);
  // (x-mean)^2
  auto numerator = rewriter.create<mlir::arith::MulFOp>(op.getLoc(), subtraction, subtraction);
  // -(x-mean)^2 / 2*variance
  auto fraction = rewriter.create<mlir::arith::MulFOp>(op.getLoc(), numerator, denominatorConst);
  // e^(-(x-mean)^2 / 2*variance)
  auto exp = rewriter.create<mlir::math::ExpOp>(op.getLoc(), fraction);
  // e^(-(x - mean)^2/2*variance)) * 1/sqrt(2*PI*variance)
  Value gaussian = rewriter.create<mlir::arith::MulFOp>(op->getLoc(), coefficientConst, exp);
  if (op.getSupportMarginal()) {
    auto isNan = rewriter.create<mlir::arith::CmpFOp>(op->getLoc(), arith::CmpFPredicate::UNO, index, index);
    auto constOne = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), rewriter.getFloatAttr(resultType, 1.0));
    gaussian = rewriter.create<mlir::arith::SelectOp>(op.getLoc(), isNan, constOne, gaussian);
  }
  rewriter.replaceOp(op, gaussian);
  return success();
}

mlir::LogicalResult mlir::spn::GaussianLogLowering::matchAndRewrite(mlir::spn::low::SPNGaussianLeaf op,
                                                                    OpAdaptor adaptor,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
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
    index = rewriter.create<mlir::arith::UIToFPOp>(op->getLoc(), resultType, index);
  } else if (auto floatIndexType = indexType.dyn_cast<FloatType>()) {
    // Widden or narrow the index floating-point type to the result floating-point type.
    if (floatIndexType.getWidth() < resultType.getWidth()) {
      index = rewriter.create<mlir::arith::ExtFOp>(op.getLoc(), resultType, index);
    } else if (floatIndexType.getWidth() > resultType.getWidth()) {
      index = rewriter.create<mlir::arith::TruncIOp>(op.getLoc(), resultType, index);
    }
  } else {
    // The input is neither float nor integer/index, fail this pattern because no conversion is possible.
    return rewriter.notifyMatchFailure(op, "Match failed because input is neither float nor integer/index");
  }

  // Calculate Gaussian distribution using the logarithm of the PDF of the Normal (Gaussian) distribution,
  // given as '-ln(stddev) - 1/2 ln(2*pi) - (x - mean)^2 / 2*stddev^2'
  // First term, -ln(stddev)
  double firstTerm = -log(op.getStddev().convertToDouble());
  // Second term, - 1/2 ln(2*pi)
  double secondTerm = -0.5 * log(2 * M_PI);
  // Denominator, - 1/2*(stddev^2)
  double denominator = -(1.0 / (2.0 * op.getStddev().convertToDouble() * op.getStddev().convertToDouble()));
  auto denominatorConst = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(),
                                                            rewriter.getFloatAttr(resultType, denominator));
  // Coefficient, summing up the first two constant terms
  double coefficient = firstTerm + secondTerm;
  auto coefficientConst = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(),
                                                            rewriter.getFloatAttr(resultType, coefficient));
  // x - mean
  auto meanConst = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(),
                                                     rewriter.getFloatAttr(resultType,
                                                                           op.getMeanAttr().getValueAsDouble()));
  auto subtraction = rewriter.create<mlir::arith::SubFOp>(op.getLoc(), index, meanConst);
  // (x-mean)^2
  auto numerator = rewriter.create<mlir::arith::MulFOp>(op.getLoc(), subtraction, subtraction);
  // - ( (x-mean)^2 / 2 * stddev^2 )
  auto fraction = rewriter.create<mlir::arith::MulFOp>(op.getLoc(), numerator, denominatorConst);
  // -ln(stddev) - 1/2 ln(2*pi) - 1/2*(stddev^2) * (x - mean)^2
  Value gaussian = rewriter.create<mlir::arith::AddFOp>(op->getLoc(), coefficientConst, fraction);
  if (op.getSupportMarginal()) {
    auto isNan = rewriter.create<mlir::arith::CmpFOp>(op->getLoc(), arith::CmpFPredicate::UNO, index, index);
    auto constOne = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), rewriter.getFloatAttr(resultType, 0.0));
    gaussian = rewriter.create<mlir::arith::SelectOp>(op.getLoc(), isNan, constOne, gaussian);
  }
  rewriter.replaceOp(op, gaussian);
  return success();
}

namespace {

  template<typename SourceOp>
  mlir::LogicalResult replaceOpWithGlobalMemref(SourceOp op, mlir::ConversionPatternRewriter& rewriter,
                                                mlir::Value indexOperand, llvm::ArrayRef<mlir::Attribute> arrayValues,
                                                mlir::Type resultType, const std::string& tablePrefix,
                                                bool computesLog) {
    static int tableCount = 0;
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
    auto symbolName = tablePrefix + std::to_string(tableCount++);
    auto visibility = rewriter.getStringAttr("private");
    auto memrefType = mlir::MemRefType::get({(long) arrayValues.size()}, resultType);
    (void) rewriter.create<mlir::memref::GlobalOp>(op.getLoc(), symbolName, visibility,
                                                   memrefType, valArrayAttr, true, mlir::IntegerAttr());
    // Restore insertion point
    rewriter.restoreInsertionPoint(restore);

    // Use GetGlobalMemref operation to access the global created above.
    auto addressOf = rewriter.template create<mlir::memref::GetGlobalOp>(op.getLoc(), memrefType, symbolName);
    // Convert input value from float to integer if necessary.
    mlir::Value index = indexOperand;
    if (!index.getType().isIntOrIndex()) {
      // If the input type is not an integer, but also not a float, we cannot convert it and this pattern fails.
      if (!index.getType().isIntOrFloat()) {
        return mlir::failure();
      }
      index = rewriter.template create<mlir::arith::FPToUIOp>(op.getLoc(), rewriter.getI64Type(), index);
    }
    // Cast input value to index if necessary.
    if (!index.getType().isIndex()) {
      index = rewriter.template create<mlir::arith::IndexCastOp>(op.getLoc(), rewriter.getIndexType(), index);
    }
    // Replace the source operation with a load from the global memref,
    // using the source operation's input value as index.
    mlir::Value leaf = rewriter.template create<mlir::memref::LoadOp>(op.getLoc(), addressOf, mlir::ValueRange{index});
    if (op.getSupportMarginal()) {
      assert(indexOperand.getType().template isa<mlir::FloatType>());
      auto isNan = rewriter.create<mlir::arith::CmpFOp>(op->getLoc(), mlir::arith::CmpFPredicate::UNO,
                                                 indexOperand, indexOperand);
      auto marginalValue = (computesLog) ? 0.0 : 1.0;
      auto constOne = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(),
                                                        rewriter.getFloatAttr(resultType, marginalValue));
      leaf = rewriter.create<mlir::arith::SelectOp>(op.getLoc(), isNan, constOne, leaf);
    }
    rewriter.replaceOp(op, leaf);
    return mlir::success();
  }

}

mlir::LogicalResult mlir::spn::HistogramLowering::matchAndRewrite(mlir::spn::low::SPNHistogramLeaf op,
                                                                  OpAdaptor adaptor,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  // Check for single operand, i.e. the index value.
  assert(operands.size() == 1);

  // Collect all mappings from input var value to probability value in a map
  // and compute the minimum lower bound & maximum upper bound.
  llvm::DenseMap<int, double> values;
  int minLB = std::numeric_limits<int>::max();
  int maxUB = std::numeric_limits<int>::min();
  for (auto& b : op.getBucketsAttr()) {
    low::BucketAttr bucket = b.cast<low::BucketAttr>();
    auto lb = bucket.getLb();
    auto ub = bucket.getUb();
    auto val = bucket.getVal().convertToDouble();
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

  Type resultType = op.getResult().getType();
  bool computesLog = false;
  if (auto logType = resultType.dyn_cast<low::LogType>()) {
    resultType = logType.getBaseType();
    computesLog = true;
  }
  if (!resultType.isIntOrFloat()) {
    // Currently only handling Int and Float result types.
    return failure();
  }

  // Flatten the map into an array by filling up empty indices with 0 values.
  SmallVector<Attribute, 256> valArray;
  for (int i = 0; i < maxUB; ++i) {
    double indexVal = NAN;
    if (values.count(i)) {
      indexVal = (computesLog) ? log(values[i]) : values[i];
    } else {
      // Fill up with 0 if no value was defined by the histogram.
      indexVal = (computesLog) ? static_cast<double>(-INFINITY) : 0;
    }
    // Construct attribute with constant value. Need to distinguish cases here due to different builder methods.
    if (resultType.isIntOrIndex()) {
      valArray.push_back(rewriter.getIntegerAttr(resultType, (int) indexVal));
    } else {
      valArray.push_back(rewriter.getFloatAttr(resultType, indexVal));
    }
  }

  return replaceOpWithGlobalMemref<low::SPNHistogramLeaf>(op, rewriter, operands[0], valArray,
                                                          resultType, "histogram_", computesLog);
}
mlir::LogicalResult mlir::spn::CategoricalLowering::matchAndRewrite(mlir::spn::low::SPNCategoricalLeaf op,
                                                                    OpAdaptor adaptor,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  // Check for single operand, i.e., the index value.
  assert(operands.size() == 1);
  Type resultType = op.getResult().getType();
  bool computesLog = false;
  if (auto logType = resultType.dyn_cast<low::LogType>()) {
    resultType = logType.getBaseType();
    computesLog = true;
  }
  SmallVector<Attribute, 5> values;
  for (auto val : op.getProbabilities().getValue()) {
    if (computesLog) {
      auto floatVal = val.dyn_cast<FloatAttr>();
      assert(floatVal);
      values.push_back(FloatAttr::get(resultType, log(floatVal.getValueAsDouble())));
    } else {
      values.push_back(val);
    }
  }
  return replaceOpWithGlobalMemref<low::SPNCategoricalLeaf>(op, rewriter, operands[0],
                                                            values, resultType, "categorical_", computesLog);
}

mlir::LogicalResult mlir::spn::ResolveConvertToVector::matchAndRewrite(mlir::spn::low::SPNConvertToVector op,
                                                                       OpAdaptor adaptor,
                                                                       mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  
  assert(operands.size() == 1);
  if (operands[0].getType() == op.getResult().getType()) {
    // This handles the case the ConvertToVector was inserted as a materialization, but the input value
    // has been vectorized in the meantime, so the conversion can be trivially resolved.
    rewriter.replaceOp(op, operands);
    return success();
  }
  if (operands[0].getDefiningOp()->hasTrait<mlir::OpTrait::ConstantLike>()) {
    // This handles the case that the ConvertToVector was inserted between a
    // constant operation (typically from the Standard dialect, which cannot be marked as vectorized)
    // and its user. In this case, we can replace the ConverToVector by a vector constant,
    // using the constant value of the operand for all vector lanes.
    SmallVector<OpFoldResult, 1> foldResults;
    auto foldReturn = operands[0].getDefiningOp()->fold({}, foldResults);
    if (failed(foldReturn)) {
      return rewriter.notifyMatchFailure(op, "Failed to fold constant operand");
    }
    if (auto constAttr = foldResults.front().dyn_cast<Attribute>()) {
      auto vectorType = op.getResult().getType().dyn_cast<VectorType>();
      assert(vectorType);
      assert(vectorType.getRank() == 1 && "Expecting only 1D vectors");
      auto vectorWidth = vectorType.getDimSize(0);
      SmallVector<Attribute, 4> constantValues;
      for (int i = 0; i < vectorWidth; ++i) {
        constantValues.push_back(constAttr);
      }
      auto constVectorAttr = DenseElementsAttr::get(vectorType, constantValues);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, constVectorAttr);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "Constant folding did not yield a constant attribute");
  }
  return rewriter.notifyMatchFailure(op, "Conversion to vector cannot be resolved trivially");
}

mlir::LogicalResult mlir::spn::ResolveStripLog::matchAndRewrite(mlir::spn::low::SPNStripLog op,
                                                                OpAdaptor adaptor,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not resolve vectorized operation");
  }
  assert(operands.size() == 1);
  if (operands[0].getType() != op.getTarget()) {
    return rewriter.notifyMatchFailure(op, "Could not resolve StripLog trivially");
  }
  rewriter.replaceOp(op, operands[0]);
  return success();
}

mlir::LogicalResult mlir::spn::ResolveConvertLog::matchAndRewrite(mlir::spn::low::SPNConvertLog op,
                                                                  OpAdaptor adaptor,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  auto operands = adaptor.getOperands();
  assert(operands.size() == 1);
  auto baseType = typeConverter->convertType(op.getResult().getType());
  assert(operands[0].getType() == baseType);
  // Simply replace the conversion by its input operand. All users of the conversion should be
  // converted subsequently.
  rewriter.replaceOp(op, operands[0]);
  return success();
}
