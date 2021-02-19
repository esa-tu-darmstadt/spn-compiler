//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <mlir/IR/BlockAndValueMapping.h>
#include "LoSPNtoCPU/Vectorization/VectorizationPatterns.h"
#include "TargetInformation.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"

mlir::LogicalResult mlir::spn::VectorizeBatchTask::matchAndRewrite(mlir::spn::low::SPNTask op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  static int taskCount = 0;

  if (op.batchSize() <= 1) {
    return rewriter.notifyMatchFailure(op,
                                       "Specialized for batch vectorization, does not match for batchSize == 1");
  }

  assert(operands.back().getType().isa<MemRefType>());
  auto computationType = operands.back().getType().dyn_cast<MemRefType>().getElementType();
  auto hwVectorWidth = TargetInformation::nativeCPUTarget().getHWVectorEntries(computationType);

  if (hwVectorWidth <= 1) {
    return rewriter.notifyMatchFailure(op,
                                       llvm::formatv(
                                           "No vectorization possible for data-type {} on the requested target",
                                           computationType));
  }

  // TODO Check if all nodes can be vectorized before trying to do so.

  // Let the user know which vector width will be used.
  op->emitRemark() << "Attempting to vectorize with vector width " << hwVectorWidth
                   << " for data-type " << computationType;

  // Emit a warning if the target vector width does not divide the requested batch size.
  // This will cause a part of each batch (batchSize % vectorWidth elements) to be processed
  // by the scalar epilog loop instead of the vectorized loop.
  if ((op.batchSize() % hwVectorWidth) != 0) {
    op.emitWarning() << "The target vector width " << hwVectorWidth
                     << " does not divide the requested batch size " << op.batchSize()
                     << "; This can result in degraded performance. "
                     << "Choose the batch size as a multiple of the vector width "
                     << hwVectorWidth;
  }

  auto restore = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(op->getParentOfType<mlir::ModuleOp>().getBody());
  SmallVector<Type, 5> inputTypes;
  for (auto operand : operands) {
    inputTypes.push_back(operand.getType());
  }
  auto funcType = FunctionType::get(rewriter.getContext(), inputTypes, {});
  auto taskFunc = rewriter.create<FuncOp>(op->getLoc(), Twine("task_", std::to_string(taskCount++)).str(),
                                          funcType);
  auto taskBlock = taskFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(taskBlock);
  auto numSamples = rewriter.create<DimOp>(op.getLoc(), taskBlock->getArgument(0), 0);
  auto vectorWidthConst = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getI64IntegerAttr(hwVectorWidth));
  auto remainder = rewriter.create<mlir::UnsignedRemIOp>(op.getLoc(), numSamples, vectorWidthConst);
  auto ubVectorized = rewriter.create<mlir::SubIOp>(op.getLoc(), numSamples, remainder);

  // Create the vectorized loop, iterating from 0 to ubVectorized, in steps of hwVectorWidth.
  auto lbVectorized = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  auto ubVectorCast = rewriter.create<mlir::IndexCastOp>(op.getLoc(), ubVectorized, rewriter.getIndexType());
  auto stepVectorized = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(hwVectorWidth));
  auto vectorizedLoop = rewriter.create<mlir::scf::ForOp>(op.getLoc(), lbVectorized, ubVectorCast, stepVectorized);
  auto& vectorLoopBody = vectorizedLoop.getLoopBody().front();

  auto restoreTask = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&vectorLoopBody);
  auto oldTaskArgs = op.body().front().getArguments();
  BlockAndValueMapping mapVectorTaskArgs;
  // Map from batchIndex to vectorized loop induction var.
  mapVectorTaskArgs.map(oldTaskArgs.front(), vectorizedLoop.getInductionVar());
  int i = 1;
  for (auto bArg : taskBlock->getArguments()) {
    mapVectorTaskArgs.map(oldTaskArgs[i++], bArg);
  }
  // Copy the operations from the Task's content to the vectorized loop
  for (auto& node : op.body().front()) {
    if (isa<low::SPNReturn>(&node)) {
      continue;
    }
    auto copy = rewriter.clone(node, mapVectorTaskArgs);
    if (auto batchRead = dyn_cast<low::SPNBatchRead>(copy)) {
      batchRead.vectorFactorAttr(rewriter.getI32IntegerAttr(hwVectorWidth));
    } else if (auto batchWrite = dyn_cast<low::SPNBatchWrite>(copy)) {
      batchWrite.vectorFactorAttr(rewriter.getI32IntegerAttr(hwVectorWidth));
    }
  }

  rewriter.restoreInsertionPoint(restoreTask);

  // Create the scalar epilog loop, iterating from ubVectorized to numSamples, in steps of 1.
  auto lbScalar = rewriter.create<mlir::IndexCastOp>(op.getLoc(), ubVectorized, rewriter.getIndexType());
  auto ubScalar = rewriter.create<mlir::IndexCastOp>(op.getLoc(), numSamples, rewriter.getIndexType());
  auto stepScalar = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1));
  auto scalarLoop = rewriter.create<mlir::scf::ForOp>(op.getLoc(), lbScalar, ubScalar, stepScalar);
  auto& scalarLoopBody = scalarLoop.getLoopBody().front();

  restoreTask = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&scalarLoopBody);
  SmallVector<Value, 5> blockReplacementArgs;
  blockReplacementArgs.push_back(scalarLoop.getInductionVar());
  for (auto bArg : taskBlock->getArguments()) {
    blockReplacementArgs.push_back(bArg);
  }
  rewriter.mergeBlockBefore(&op.body().front(), scalarLoopBody.getTerminator(), blockReplacementArgs);
  scalarLoopBody.walk([&rewriter](low::SPNReturn ret) {
    assert(ret.returnValues().empty() && "Task return should be empty");
    rewriter.eraseOp(ret);
  });

  rewriter.restoreInsertionPoint(restoreTask);
  rewriter.create<ReturnOp>(op->getLoc());
  // Insert a call to the newly created task function.
  rewriter.restoreInsertionPoint(restore);
  rewriter.replaceOpWithNewOp<mlir::CallOp>(op, taskFunc, operands);
  return success();

}

//
// Anonymous namespace holding a bunch of helper functions.
//
namespace {

  template<typename T>
  mlir::ConstantOp broadcastVectorConstant(mlir::VectorType type, T value,
                                           mlir::ConversionPatternRewriter& rewriter, mlir::Location loc) {
    assert(type.hasStaticShape());
    llvm::SmallVector<T, 8> array;
    for (int i = 0; i < type.getNumElements(); ++i) {
      array.push_back(value);
    }
    auto constAttr = mlir::DenseElementsAttr::get(type, (llvm::ArrayRef<T>) array);
    auto constValue = rewriter.create<mlir::ConstantOp>(loc, constAttr);
    return constValue;
  }

}

mlir::LogicalResult mlir::spn::VectorizeBatchRead::matchAndRewrite(mlir::spn::low::SPNBatchRead op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  // Replace the vectorized version of a BatchRead with a Gather load from the input memref.
  if (op.vectorFactor() <= 1) {
    return rewriter.notifyMatchFailure(op, "No vectorization possible for this target");
  }
  assert(operands.size() == 2);
  assert(operands[0].getType().isa<MemRefType>());
  assert(operands[1].getType().isa<IndexType>());
  auto memRef = operands[0].getType().dyn_cast<MemRefType>();
  // Assume that the second dimension (i.e., the number of features per sample is a static dimension).
  assert(!memRef.isDynamicDim(1));
  auto numFeatures = memRef.getDimSize(1);
  auto vectorType = VectorType::get({op.vectorFactor()}, op.getResult().getType());
  // Broadcast the batchIndex
  auto vectorOfIndex = VectorType::get(op.vectorFactor(), rewriter.getI32Type());
  auto batchIndex = rewriter.create<vector::BroadcastOp>(op.getLoc(), vectorOfIndex, operands[1]);
  // Create a constant vector with the offsets of the elements from the first sample.
  SmallVector<int, 4> offsets;
  for (int i = 0; i < op.vectorFactor(); ++i) {
    offsets.push_back(i * numFeatures + op.sampleIndex());
  }
  auto constAttr = mlir::DenseElementsAttr::get(vectorOfIndex, (llvm::ArrayRef<int>) offsets);
  auto constOffset = rewriter.create<ConstantOp>(op.getLoc(), constAttr);
  // Add the offsets to the base index from the batchIndex.
  auto addresses = rewriter.create<AddIOp>(op.getLoc(), batchIndex, constOffset);
  // Create constant passThru.
  auto passThru = broadcastVectorConstant(vectorType, 0.0,
                                          rewriter, op->getLoc());
  // Construct the constant mask.
  auto mask = broadcastVectorConstant(mlir::VectorType::get(op.vectorFactor(), rewriter.getI1Type()), true,
                                      rewriter, op->getLoc());
  rewriter.replaceOpWithNewOp<vector::GatherOp>(op, vectorType, operands[0], addresses, mask, passThru);
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeBatchWrite::matchAndRewrite(mlir::spn::low::SPNBatchWrite op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  if (op.vectorFactor() <= 1) {
    return rewriter.notifyMatchFailure(op, "No vectorization possible for this target");
  }
  assert(operands.size() == 3);
  VectorType vectorType;
  auto result = operands[0];
  if (!result.getType().isa<VectorType>()) {
    vectorType = VectorType::get({op.vectorFactor()}, result.getType());
    result = typeConverter->materializeTargetConversion(rewriter, op->getLoc(), vectorType, result);
    assert(result);
  } else {
    vectorType = result.getType().dyn_cast<VectorType>();
  }
  assert(operands[1].getType().dyn_cast<MemRefType>().getElementType() == vectorType.getElementType()
             && "Result type and element type of MemRef must match");
  assert(operands[1].getType().isa<MemRefType>());
  assert(operands[2].getType().isa<IndexType>());
  rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(op, result, operands[1],
                                                       operands[2]);
  op->getParentOfType<FuncOp>()->dump();
  return success();
}

mlir::LogicalResult mlir::spn::VectorizeMul::matchAndRewrite(mlir::spn::low::SPNMul op,
                                                             llvm::ArrayRef<mlir::Value> operands,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
  return OpConversionPattern::matchAndRewrite(op, operands, rewriter);
}

mlir::LogicalResult mlir::spn::VectorizeAdd::matchAndRewrite(mlir::spn::low::SPNAdd op,
                                                             llvm::ArrayRef<mlir::Value> operands,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
  return OpConversionPattern::matchAndRewrite(op, operands, rewriter);
}

mlir::LogicalResult mlir::spn::VectorizeGaussian::matchAndRewrite(mlir::spn::low::SPNGaussianLeaf op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.size() == 1 && "Expecting only a single operand for Gaussian leaf");

  auto feature = operands.front();
  operands.front().getDefiningOp()->dump();
  rewriter.getRemappedValue(operands.front()).dump();

  // Check that the operand is a vector of floats.
  // TODO Handle integer vectors via conversion in vectorized mode.
  if (!feature.getType().isa<VectorType>() ||
      !feature.getType().dyn_cast<VectorType>().getElementType().isa<FloatType>()) {
    return failure();
  }
  auto vectorType = feature.getType().dyn_cast<VectorType>();
  assert(vectorType);
  auto featureType = vectorType.getElementType().dyn_cast<FloatType>();
  assert(featureType);

  // Get the return type
  Type resultType = op.getResult().getType();
  // Check the Gaussian returns a float result.
  if (!resultType.isa<FloatType>()) {
    return failure();
  }
  auto floatResultType = resultType.dyn_cast<FloatType>();
  assert(floatResultType);

  // FPTrunc and FPExt currently do not support vector types.
  // Vectorization of a Gaussian must fail if its involves changing the width of
  // the floating type between input (feature) and result.
  if (featureType.getWidth() != floatResultType.getWidth()) {
    return rewriter.notifyMatchFailure(op,
                                       "Aborting vectorization: Cannot vectorize Gaussian leaf as the requested input type"
                                           +
                                               llvm::formatv("{}", featureType) +
                                           " cannot be converted to the data-type for computation" +
                                           llvm::formatv("{}", floatResultType) +
                                           " in vectorized mode");
  }

  // Calculate Gaussian distribution using e^(-(x - mean)^2/2*variance))/sqrt(2*PI*variance)
  // Variance from standard deviation.
  double variance = op.stddev().convertToDouble() * op.stddev().convertToDouble();
  // 1/sqrt(2*PI*variance)
  double coefficient = 1.0 / (std::sqrt(2.0 * M_PI * variance));
  auto coefficientConst = broadcastVectorConstant(vectorType, coefficient, rewriter, op.getLoc());
  // -1/(2*variance)
  double denominator = -1.0 / (2.0 * variance);
  auto denominatorConst = broadcastVectorConstant(vectorType, denominator, rewriter, op.getLoc());
  // x - mean
  auto meanConst = broadcastVectorConstant(vectorType, op.mean().convertToDouble(), rewriter, op.getLoc());
  auto subtraction = rewriter.create<mlir::SubFOp>(op.getLoc(), feature, meanConst);
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

mlir::LogicalResult mlir::spn::VectorizeCategorical::matchAndRewrite(mlir::spn::low::SPNCategoricalLeaf op,
                                                                     llvm::ArrayRef<mlir::Value> operands,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
  return OpConversionPattern::matchAndRewrite(op, operands, rewriter);
}

mlir::LogicalResult mlir::spn::VectorizeHistogram::matchAndRewrite(mlir::spn::low::SPNHistogramLeaf op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  return OpConversionPattern::matchAndRewrite(op, operands, rewriter);
}

mlir::LogicalResult mlir::spn::ResolveConvertToVector::matchAndRewrite(mlir::spn::low::SPNConvertToVector op,
                                                                       llvm::ArrayRef<mlir::Value> operands,
                                                                       mlir::ConversionPatternRewriter& rewriter) const {
  return OpConversionPattern::matchAndRewrite(op, operands, rewriter);
}
