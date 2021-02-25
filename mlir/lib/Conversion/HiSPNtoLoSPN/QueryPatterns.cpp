//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "HiSPNtoLoSPN/QueryPatterns.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"

mlir::LogicalResult mlir::spn::JointQueryLowering::matchAndRewrite(mlir::spn::high::JointQuery op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  //
  // Translate the JointQuery into a SPNKernel. The Kernel takes a 2D tensor as input,
  // with the first dimension equal to the batchSize (dynamic for size > 1) and
  // the second dimension equal to the number of features inside a single sample.
  // It produces a single 1D tensor with batchSize (dynamic for size > 1) many elements.
  //
  // The result of a JointQuery is converted to log before returning. Currently, F64 is always used to
  // represent the log-result.
  auto compType = rewriter.getF64Type();
  auto dynamicBatchSize = (op.getBatchSize() == 1) ? 1 : -1;
  // 1 x numFeatures x inputType for batchSize == 1, ? x numFeatures x inputType else.
  auto inputType = RankedTensorType::get({dynamicBatchSize, op.getNumFeatures()}, op.getFeatureDataType());
  // 1 x compType for batchSize == 1, ? x compType else.
  auto resultType = RankedTensorType::get({dynamicBatchSize}, compType);
  // Create the function type of the kernel.
  auto kernelType = FunctionType::get(rewriter.getContext(), TypeRange{inputType}, TypeRange{resultType});
  auto kernel = rewriter.create<low::SPNKernel>(op.getLoc(), op.kernelName(), kernelType);
  auto kernelBlock = kernel.addEntryBlock();
  rewriter.setInsertionPointToStart(kernelBlock);
  // Create a single task inside the kernel, taking the same arguments and producing the same
  // result as the kernel.
  auto task = rewriter.create<low::SPNTask>(op.getLoc(), TypeRange{resultType},
                                            kernelBlock->getArgument(0),
                                            op.getBatchSize());
  auto restoreKernel = rewriter.saveInsertionPoint();
  // The block of the task has another argument for the batch index as first argument.
  auto taskBlock = task.addEntryBlock();
  rewriter.setInsertionPointToStart(taskBlock);
  //
  // Generate a SPNBatchIndexRead for every feature of the input sample.
  auto batchIndex = task.getBatchIndex();
  auto inputArg = taskBlock->getArgument(1);
  SmallVector<Value> inputValues;
  SmallVector<Type> inputTypes;
  for (unsigned i = 0; i < op.numFeatures(); ++i) {
    auto arg = rewriter.create<low::SPNBatchExtract>(op.getLoc(), op.getFeatureDataType(),
                                                     inputArg, batchIndex, i);
    inputValues.push_back(arg);
    inputTypes.push_back(arg.getType());
  }
  // Create the body of the taks, using the features values as arguments.
  auto body = rewriter.create<low::SPNBody>(op.getLoc(), TypeRange{compType},
                                            inputValues);
  auto restoreTask = rewriter.saveInsertionPoint();
  auto bodyBlock = rewriter.createBlock(&body.getRegion(), {}, inputTypes);
  //
  // Merge the content of the DAG (which has been lowered to LoSPN in a previous step) to
  // the body of the task.
  auto spnDAG = dyn_cast<high::Graph>(op.graph().front().front());
  assert(spnDAG && "Expecting the first operation to be the SPN DAG");
  rewriter.mergeBlocks(&spnDAG.graph().front(), bodyBlock, bodyBlock->getArguments());
  rewriter.restoreInsertionPoint(restoreTask);
  // Create a SPNBatchCollect for the result produced by the body, terminating the task.
  auto output = rewriter.create<low::SPNBatchCollect>(op.getLoc(), resultType,
                                                      ValueRange{body.getResult(0)},
                                                      batchIndex);
  rewriter.create<low::SPNReturn>(op.getLoc(), output.tensors());
  rewriter.restoreInsertionPoint(restoreKernel);
  rewriter.create<low::SPNReturn>(op.getLoc(), task.getResult(0));
  // Erase the original JointQuery that is now represented by the SPNKernel.
  rewriter.eraseOp(op);
  return success();
}
