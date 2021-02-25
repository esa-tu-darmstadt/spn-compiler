//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/NodePatterns.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "LoSPN/LoSPNAttributes.h"

mlir::LogicalResult mlir::spn::BatchReadLowering::matchAndRewrite(mlir::spn::low::SPNBatchRead op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  // Replace the BatchRead with a scalar load from the input memref,
  // using the batchIndex and the constant sample index.
  assert(operands.size() == 2 && "Expecting two operands for BatchRead");
  assert(operands[0].getType().isa<MemRefType>());
  assert(operands[1].getType().isa<IndexType>());
  auto constSampleIndex = rewriter.create<ConstantOp>(op->getLoc(), rewriter.getIndexAttr(op.sampleIndex()));
  rewriter.replaceOpWithNewOp<LoadOp>(op, operands[0], ValueRange{operands[1], constSampleIndex});
  return success();
}

mlir::LogicalResult mlir::spn::BatchWriteLowering::matchAndRewrite(mlir::spn::low::SPNBatchWrite op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  // Replace the BatchWrite with a store to the input memref,
  // using the batchIndex.
  assert(operands.size() == 3 && "Expecting three operands for BatchWrite");
  assert(operands[1].getType().isa<MemRefType>());
  assert(operands[1].getType().dyn_cast<MemRefType>().getElementType() == operands[0].getType()
             && "Result type and element type of MemRef must match");
  assert(operands[2].getType().isa<IndexType>());
  rewriter.replaceOpWithNewOp<StoreOp>(op, operands[0], operands[1], operands[2]);
  return success();
}

mlir::LogicalResult mlir::spn::CopyLowering::matchAndRewrite(mlir::spn::low::SPNCopy op,
                                                             llvm::ArrayRef<mlir::Value> operands,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.size() == 2 && "Expecting two operands for Copy");
  assert(operands[0].getType().isa<MemRefType>());
  assert(operands[1].getType().isa<MemRefType>());
  rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, operands[0], operands[1]);
  return success();
}

// Anonymous namespace holding helper functions.
mlir::LogicalResult mlir::spn::ConstantLowering::matchAndRewrite(mlir::spn::low::SPNConstant op,
                                                                 llvm::ArrayRef<mlir::Value> operands,
                                                                 mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.empty() && "Expecting no operands for Constant");
  rewriter.replaceOpWithNewOp<ConstantOp>(op, op.valueAttr());
  return success();
}

mlir::LogicalResult mlir::spn::ReturnLowering::matchAndRewrite(mlir::spn::low::SPNReturn op,
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

mlir::LogicalResult mlir::spn::LogLowering::matchAndRewrite(mlir::spn::low::SPNLog op,
                                                            llvm::ArrayRef<mlir::Value> operands,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.size() == 1 && "Expecting one operand for Log");
  rewriter.replaceOpWithNewOp<math::LogOp>(op, operands[0]);
  return success();
}

mlir::LogicalResult mlir::spn::MulLowering::matchAndRewrite(mlir::spn::low::SPNMul op,
                                                            llvm::ArrayRef<mlir::Value> operands,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.size() == 2 && "Expecting two operands for Mul");
  if (!operands[0].getType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Currently only matches floating-point multiplications");
  }
  rewriter.replaceOpWithNewOp<MulFOp>(op, operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::AddLowering::matchAndRewrite(mlir::spn::low::SPNAdd op,
                                                            llvm::ArrayRef<mlir::Value> operands,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.size() == 2 && "Expecting two operands for Add");
  if (!operands[0].getType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Currently only matches floating-point additions");
  }
  rewriter.replaceOpWithNewOp<AddFOp>(op, operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::GaussianLowering::matchAndRewrite(mlir::spn::low::SPNGaussianLeaf op,
                                                                 llvm::ArrayRef<mlir::Value> operands,
                                                                 mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.size() == 1 && "Expecting a single operands for Gaussian");
  if (!operands.front().getType().isIntOrFloat()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
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
  rewriter.replaceOpWithNewOp<mlir::MulFOp>(op, coefficientConst, exp);
  return success();
}

namespace {

  template<typename SourceOp>
  mlir::LogicalResult replaceOpWithGlobalMemref(SourceOp op, mlir::ConversionPatternRewriter& rewriter,
                                                mlir::Value indexOperand, llvm::ArrayRef<mlir::Attribute> arrayValues,
                                                const std::string& tablePrefix) {
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
    auto symbolName = tablePrefix + std::to_string(tableCount++);
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

mlir::LogicalResult mlir::spn::HistogramLowering::matchAndRewrite(mlir::spn::low::SPNHistogramLeaf op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
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
  for (auto& b : op.bucketsAttr()) {
    auto bucket = b.cast<low::Bucket>();
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

  return replaceOpWithGlobalMemref<low::SPNHistogramLeaf>(op, rewriter, operands[0], valArray,
                                                          "histogram_");
}
mlir::LogicalResult mlir::spn::CategoricalLowering::matchAndRewrite(mlir::spn::low::SPNCategoricalLeaf op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  // Check for single operand, i.e., the index value.
  assert(operands.size() == 1);

  return replaceOpWithGlobalMemref<low::SPNCategoricalLeaf>(op, rewriter, operands[0],
                                                            op.probabilitiesAttr().getValue(),
                                                            "categorical_");
}
