//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "HiSPNtoLoSPN/QueryPatterns.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/IR/BuiltinTypes.h"

mlir::LogicalResult mlir::spn::JointQueryLowering::matchAndRewrite(
    mlir::spn::high::JointQuery op,
    mlir::spn::high::JointQuery::Adaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  //
  // Translate the JointQuery into a SPNKernel. The Kernel takes a 2D tensor as
  // input, with the first dimension equal to the batchSize (dynamic for size >
  // 1) and the second dimension equal to the number of features inside a single
  // sample. It produces a single 1D tensor with batchSize (dynamic for size >
  // 1) many elements.
  //
  Type compType = typeConverter->convertType(
      high::ProbabilityType::get(rewriter.getContext()));
  if (auto logType = compType.dyn_cast<low::LogType>()) {
    // If the computation is performed in log-space, the LogType wraps the type
    // that is actually used to perform arithmetic to flag log-computation.
    // Outside of the body itself, we still use the base-type, as the LogType
    // is not a legal element type of MemRef, causing problem in bufferization.
    // The return of the body is the place where the "conversion" will take
    // place, but no actual conversion is required.
    compType = logType.getBaseType();
  }
  auto dynamicBatchSize = ShapedType::kDynamic;
  // 1 x numFeatures x inputType for batchSize == 1, ? x numFeatures x inputType
  // else.
  auto inputType = RankedTensorType::get(
      {dynamicBatchSize, op.getNumFeatures()}, op.getFeatureDataType());
  // 1 x compType for batchSize == 1, ? x compType else.
  auto resultType = RankedTensorType::get({1, dynamicBatchSize}, compType);
  // Create the function type of the kernel.
  auto kernelType = FunctionType::get(
      rewriter.getContext(), TypeRange{inputType}, TypeRange{resultType});
  auto kernel = rewriter.create<low::SPNKernel>(op.getLoc(), op.getQueryName(),
                                                kernelType);
  auto kernelBlock = &kernel.getBlocks().front();
  rewriter.setInsertionPointToStart(kernelBlock);
  // Create a single task inside the kernel, taking the same arguments and
  // producing the same result as the kernel.
  auto task = rewriter.create<low::SPNTask>(op.getLoc(), TypeRange{resultType},
                                            kernelBlock->getArgument(0),
                                            op.getBatchSize());
  auto restoreKernel = rewriter.saveInsertionPoint();
  // The block of the task has another argument for the batch index as first
  // argument.
  auto taskBlock = task.addEntryBlock();
  rewriter.setInsertionPointToStart(taskBlock);
  //
  // Generate a SPNBatchIndexRead for every feature of the input sample.
  auto batchIndex = task.getBatchIndex();
  auto inputArg = taskBlock->getArgument(1);
  SmallVector<Value> inputValues;
  SmallVector<Type> inputTypes;
  for (unsigned i = 0; i < op.getNumFeatures(); ++i) {
    auto arg = rewriter.create<low::SPNBatchExtract>(
        op.getLoc(), op.getFeatureDataType(), inputArg, batchIndex, i,
        rewriter.getBoolAttr(false));
    inputValues.push_back(arg);
    inputTypes.push_back(arg.getType());
  }
  // Create the body of the taks, using the features values as arguments.
  auto body = rewriter.create<low::SPNBody>(op.getLoc(), TypeRange{compType},
                                            inputValues);
  auto restoreTask = rewriter.saveInsertionPoint();
  auto bodyBlock = body.addEntryBlock();
  rewriter.setInsertionPointToStart(bodyBlock);
  //
  // Merge the content of the DAG (which has been lowered to LoSPN in a previous
  // step) to the body of the task.
  auto spnDAG = dyn_cast<high::Graph>(op.getGraph().front().front());
  assert(spnDAG && "Expecting the first operation to be the SPN DAG");
  rewriter.mergeBlocks(&spnDAG.getGraph().front(), bodyBlock,
                       bodyBlock->getArguments());
  rewriter.restoreInsertionPoint(restoreTask);
  // Create a SPNBatchCollect for the result produced by the body, terminating
  // the task.
  auto output = rewriter.create<low::SPNBatchCollect>(
      op.getLoc(), resultType, batchIndex, ValueRange{body.getResult(0)},
      rewriter.getBoolAttr(true));
  rewriter.create<low::SPNReturn>(op.getLoc(), output.getResult());
  rewriter.restoreInsertionPoint(restoreKernel);
  rewriter.create<low::SPNReturn>(op.getLoc(), task.getResult(0));
  // Erase the original JointQuery that is now represented by the SPNKernel.
  rewriter.eraseOp(op);
  return success();
}
