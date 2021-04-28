//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/IR/BuiltinOps.h"
#include "SPNtoStandard/SPNtoStandardPatterns.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "../../../LoSPNtoCPU/Target/TargetInformation.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "SPNtoStandard/Vectorization/BatchVectorizationPatterns.h"
#include <cmath>

/*
 * Bunch of helper functions declared in an anonymous namespace.
 */
namespace {

  std::pair<mlir::Value, mlir::Value> combineHalves(mlir::Value& leftIn, mlir::Value& rightIn,
                                                    unsigned vectorSize, unsigned step,
                                                    mlir::ConversionPatternRewriter& rewriter,
                                                    mlir::Location loc) {
    unsigned leftIndex = 0;
    unsigned rightIndex = vectorSize;
    llvm::SmallVector<int64_t, 8> firstPermutation;
    for (unsigned i = 0; i < vectorSize / (step * 2); ++i) {
      for (unsigned k = 0; k < step; ++k) {
        firstPermutation.push_back(leftIndex++);
      }
      for (unsigned k = 0; k < step; ++k) {
        firstPermutation.push_back(rightIndex++);
      }
    }
    llvm::SmallVector<int64_t, 8> secondPermutation;
    for (unsigned i = 0; i < vectorSize / (step * 2); ++i) {
      for (unsigned k = 0; k < step; ++k) {
        secondPermutation.push_back(leftIndex++);
      }
      for (unsigned k = 0; k < step; ++k) {
        secondPermutation.push_back(rightIndex++);
      }
    }

    auto leftPermutation = rewriter.create<mlir::vector::ShuffleOp>(loc, leftIn, rightIn, firstPermutation);
    auto rightPermutation = rewriter.create<mlir::vector::ShuffleOp>(loc, leftIn, rightIn, secondPermutation);
    return {leftPermutation, rightPermutation};
  }

  llvm::SmallVector<mlir::Value, 8> transposeByPermutation(llvm::ArrayRef<mlir::Value> loadedVectors,
                                                           unsigned vectorSize,
                                                           mlir::ConversionPatternRewriter& rewriter,
                                                           mlir::Location loc) {
    llvm::SmallVector<mlir::Value, 8> vectors;
    for (auto v : loadedVectors) {
      vectors.push_back(v);
    }
    unsigned numPermutationStage = static_cast<unsigned>(log2(vectorSize));
    for (unsigned i = 0; i < numPermutationStage; ++i) {
      unsigned distance = 1 << i;
      llvm::SmallVector<mlir::Value, 8> newVectors;
      unsigned index = 0;
      for (unsigned j = 0; j < vectorSize / (distance * 2); ++j) {
        // Work on some elements
        for (unsigned k = 0; k < distance; ++k) {
          auto leftIn = vectors[index];
          auto rightIn = vectors[index + distance];
          auto outVec = combineHalves(leftIn, rightIn, vectorSize, distance, rewriter, loc);
          newVectors.push_back(outVec.first);
          newVectors.push_back(outVec.second);
          ++index;
        }
        // Skip some elements
        index += distance;
      }
      vectors = std::move(newVectors);
    }
    return vectors;
  }

}

/*
 * Lowering of a joint probability query with batchSize > 1 to a data-parallel, vectorized loop.
 * The vectorized loop is complemented by a scalar loop taking care of the remaining elements, in case
 * the number of samples cannot be divided by the vectorization width without remainder.
 * The lowering of the query will result in a function similar to the following pseudo-code:
 * void kernel_func(int num_samples, input_t* inputs, output_t* outputs){
 *     int epilog = num_samples % VECTOR_WIDTH;
 *     int ubVectorized = num_samples - epilog;
 *     // Vectorized loop:
 *     for(int i=0; i < ubVectorized; i += VECTOR_WIDTH){
 *       // Vectorized loop
 *     }
 *     // Epilog loop:
 *     for(int i=ubVectorized; i < num_samples; ++i){
 *       // Scalar loop
 *     }
 * }
 */
mlir::LogicalResult mlir::spn::BatchVectorizeJointLowering::matchAndRewrite(mlir::spn::JointQuery op,
                                                                            llvm::ArrayRef<mlir::Value> operands,
                                                                            mlir::ConversionPatternRewriter& rewriter) const {
  // This lowering is specialized for batch evaluation, reject queries with batch size <= 1.
  auto query = dyn_cast<QueryInterface>(op.getOperation());
  unsigned batchSize = query.getBatchSize();
  if (batchSize <= 1) {
    return failure();
  }

  auto compType = query.getComputationDataType();
  auto hwVectorWidth = TargetInformation::nativeCPUTarget().getHWVectorEntries(compType);
  // Check if the hwVectorWidth is greater than one and fail otherwise.
  if (hwVectorWidth == 1) {
    op.emitWarning() << "No vectorization possible for data-type "
                     << query.getComputationDataType() << " on the requested target";
    return failure();
  }
  // Let the user know which vector width will be used.
  op->emitRemark() << "Attempting to vectorize with vector width " << hwVectorWidth
                   << " for data-type " << query.getComputationDataType();

  // Emit a warning if the target vector width does not divide the requested batch size.
  // This will cause a part of each batch (batchSize % vectorWidth elements) to be processed
  // by the scalar epilog loop instead of the vectorized loop.
  if ((batchSize % hwVectorWidth) != 0) {
    op.emitWarning() << "The target vector width " << hwVectorWidth
                     << " does not divide the requested batch size " << batchSize
                     << "; This can result in degraded performance. Choose the batch size as a multiple of the vector width "
                     << hwVectorWidth;
  }
  // Note: The vectorized lowerings for the different leaf nodes automatically check for data-type conversions
  // that cannot be performed in vectorized mode and will warn about this and abort the vectorization by
  // by failing the lowering pattern.

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
  auto storeArg = replaceFunc.getArgument(2);
  assert(inputArg.getType().isa<MemRefType>());

  auto vectorInputType = VectorType::get({hwVectorWidth}, op.inputType());
  auto vectorType = VectorType::get({hwVectorWidth}, compType);

  auto vectorWidthConst = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getI64IntegerAttr(hwVectorWidth));
  auto remainder = rewriter.create<mlir::UnsignedRemIOp>(op.getLoc(), numSamplesArg, vectorWidthConst);
  auto ubVectorized = rewriter.create<mlir::SubIOp>(op.getLoc(), numSamplesArg, remainder);

  // Calculate the number of elements that can be loaded in a vectorized fashion.
  auto vectorLoadElements = query.getNumFeatures() - (query.getNumFeatures() % hwVectorWidth);

  // Create the vectorized loop, iterating from 0 to ubVectorized, in steps of hwVectorWidth.
  auto lbVectorized = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  auto ubVectorCast = rewriter.create<mlir::IndexCastOp>(op.getLoc(), ubVectorized, rewriter.getIndexType());
  auto stepVectorized = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(hwVectorWidth));
  auto vectorizedLoop = rewriter.create<mlir::scf::ForOp>(op.getLoc(), lbVectorized, ubVectorCast, stepVectorized);
  auto& vectorLoopBody = vectorizedLoop.getLoopBody().front();

  // Fill the vectorized loop.
  auto restore = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&vectorLoopBody);
  SmallVector<Value, 10> replaceArgs;

  /*
   * Load the first n elements as vectors, where n is determined by ceil(numFeatures / hwVectorWidth).
   * Load vector-width elements from the current first vector-width rows and
   * transpose this vector-width x vector-width matrix using shuffles, resulting in vector-width many vectors.
   */
  unsigned vectorsPerRow = vectorLoadElements / hwVectorWidth;
  SmallVector<Value, 8> rowVectors;
  for (unsigned i = 0; i < vectorsPerRow; ++i) {
    auto vectorIndex = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i * hwVectorWidth));
    SmallVector<Value, 8> vectors;
    for (unsigned k = 0; k < hwVectorWidth; ++k) {
      auto sampleIndex = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(k));
      auto loadIndex = rewriter.create<mlir::AddIOp>(op.getLoc(), vectorizedLoop.getInductionVar(), sampleIndex);
      auto kVector = rewriter.create<mlir::vector::TransferReadOp>(op.getLoc(), vectorInputType,
                                                                   (Value) inputArg,
                                                                   ValueRange{loadIndex, vectorIndex});
      vectors.push_back(kVector);
    }
    auto transposedVectors = transposeByPermutation(vectors, hwVectorWidth, rewriter, op.getLoc());
    for (auto v : transposedVectors) {
      replaceArgs.push_back(v);
    }
  }
  /*
   * Load the remaining input elements. These have to be loaded with scalar loads and inserted into the vector
   * one-by-one.
   */
  unsigned startIndex = vectorLoadElements;
  for (unsigned i = 0; i < (query.getNumFeatures() % hwVectorWidth); ++i) {
    auto alloca = rewriter.create<mlir::AllocaOp>(op.getLoc(), MemRefType::get({1}, vectorInputType));
    auto constant1 = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
    auto loadInitial = rewriter.create<mlir::LoadOp>(op.getLoc(), alloca, ValueRange{constant1});
    Value vector = loadInitial;
    auto featureIndex = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(startIndex + i));
    for (unsigned k = 0; k < hwVectorWidth; ++k) {
      auto sampleIndex = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(k));
      auto loadIndex = rewriter.create<mlir::AddIOp>(op.getLoc(), vectorizedLoop.getInductionVar(), sampleIndex);
      auto loadScalar = rewriter.create<mlir::LoadOp>(op.getLoc(), inputArg, ValueRange{loadIndex, featureIndex});
      vector = rewriter.create<mlir::vector::InsertOp>(op.getLoc(), loadScalar, vector, ArrayRef<int64_t>{k});
    }
    replaceArgs.push_back(vector);
  }
  BlockAndValueMapping argMapper;
  for (size_t i = 0; i < op.getRegion().front().getNumArguments(); ++i) {
    argMapper.map(op.getRegion().front().getArgument(i), replaceArgs[i]);
  }

  Value result;
  for (auto& o : op.getRegion().front().getOperations()) {
    if (auto retOp = dyn_cast<mlir::spn::ReturnOp>(o)) {
      // Do not copy over the SPN ReturnOp, but replace it with log + transfer write (store).
      result = rewriter.create<mlir::math::LogOp>(op.getLoc(), vectorType, argMapper.lookup(retOp.retValue().front()));
      rewriter.create<mlir::vector::TransferWriteOp>(op.getLoc(), result, (Value) storeArg,
                                                     ValueRange{vectorizedLoop.getInductionVar()});
    } else {
      rewriter.clone(o, argMapper);
    }
  }

  rewriter.restoreInsertionPoint(restore);

  // Create the scalar epilog loop, iterating from ubVectorized to numSamples, in steps of 1.
  auto lbScalar = rewriter.create<mlir::IndexCastOp>(op.getLoc(), ubVectorized, rewriter.getIndexType());
  auto ubScalar = rewriter.create<mlir::IndexCastOp>(op.getLoc(), numSamplesArg, rewriter.getIndexType());
  auto stepScalar = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1));
  auto scalarLoop = rewriter.create<mlir::scf::ForOp>(op.getLoc(), lbScalar, ubScalar, stepScalar);
  auto& scalarLoopBody = scalarLoop.getLoopBody().front();

  restore = rewriter.saveInsertionPoint();

  rewriter.setInsertionPointToStart(&scalarLoopBody);
  SmallVector<Value, 10> blockArgsReplacement;
  for (unsigned i = 0; i < op.getNumFeatures(); ++i) {
    SmallVector<Value, 2> indices;
    indices.push_back(scalarLoop.getInductionVar());
    indices.push_back(rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i)));
    auto load = rewriter.create<mlir::LoadOp>(op.getLoc(), inputArg, indices);
    blockArgsReplacement.push_back(load);
  }
  // Transfer SPN graph from JointQuery to for-loop body.
  rewriter.mergeBlockBefore(&op.getRegion().front(), scalarLoopBody.getTerminator(), blockArgsReplacement);
  // Apply logarithm to result before storing it
  rewriter.setInsertionPoint(scalarLoopBody.getTerminator());
  auto logResult = rewriter.create<mlir::math::LogOp>(op.getLoc(), graphResult.retValue().front());
  // Store the log-result to the output pointer.
  SmallVector<Value, 1> indices;
  indices.push_back(scalarLoop.getInductionVar());
  rewriter.create<mlir::StoreOp>(op.getLoc(), logResult,
                                 storeArg, indices);

  rewriter.restoreInsertionPoint(restore);
  rewriter.create<mlir::ReturnOp>(op.getLoc());
  replaceFunc.dump();
  rewriter.eraseOp(graphResult);
  rewriter.eraseOp(op);
  return success();

}

