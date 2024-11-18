//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_STRUCTUREPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_STRUCTUREPATTERNS_H

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace spn {

struct KernelLowering : OpConversionPattern<low::SPNKernel> {

  using OpConversionPattern<low::SPNKernel>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNKernel op, low::SPNKernel::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct TaskLoweringOptions {
  /// A function that builds the operation that represents the task.
  std::function<Operation *(OpBuilder &builder, Location loc, std::string name,
                            FunctionType funcType)>
      buildTask;

  /// A function that builds the call to the task operation. The original task
  /// will get replaced by this call.
  std::function<Operation *(OpBuilder &builder, Location loc, Operation *taskOp,
                            ValueRange operands)>
      buildTaskCall;
};

struct BatchTaskLowering : OpConversionPattern<low::SPNTask> {
  TaskLoweringOptions options;
  BatchTaskLowering(const TypeConverter &typeConverter, MLIRContext *context,
                    TaskLoweringOptions &options)
      : OpConversionPattern<low::SPNTask>(typeConverter, context),
        options(options) {}

  LogicalResult
  matchAndRewrite(low::SPNTask op, low::SPNTask::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct SingleTaskLowering : OpConversionPattern<low::SPNTask> {
  TaskLoweringOptions options;
  SingleTaskLowering(const TypeConverter &typeConverter, MLIRContext *context,
                     TaskLoweringOptions &options)
      : OpConversionPattern<low::SPNTask>(typeConverter, context),
        options(options) {}

  LogicalResult
  matchAndRewrite(low::SPNTask op, low::SPNTask::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct BodyLowering : OpConversionPattern<low::SPNBody> {

  using OpConversionPattern<low::SPNBody>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNBody op, low::SPNBody::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

static inline void
populateLoSPNtoCPUStructurePatterns(RewritePatternSet &patterns,
                                    MLIRContext *context,
                                    TypeConverter &typeConverter) {
  patterns.insert<KernelLowering>(typeConverter, context);
  patterns.insert<BodyLowering>(typeConverter, context);
}

void populateLoSPNtoCPUTaskPatterns(RewritePatternSet &patterns,
                                    MLIRContext *context,
                                    TypeConverter &typeConverter);
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_STRUCTUREPATTERNS_H
