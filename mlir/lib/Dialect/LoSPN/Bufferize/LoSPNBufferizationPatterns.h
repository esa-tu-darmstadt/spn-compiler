//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_LIB_DIALECT_LOSPN_BUFFERIZE_LOSPNBUFFERIZATIONPATTERNS_H
#define SPNC_MLIR_LIB_DIALECT_LOSPN_BUFFERIZE_LOSPNBUFFERIZATIONPATTERNS_H

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace spn {
namespace low {

struct KernelBufferize : OpConversionPattern<SPNKernel> {

  using OpConversionPattern<SPNKernel>::OpConversionPattern;
  using OpConversionPattern<SPNKernel>::OpAdaptor;

  LogicalResult
  matchAndRewrite(SPNKernel op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct TaskBufferize : OpConversionPattern<SPNTask> {

  using OpConversionPattern<SPNTask>::OpConversionPattern;
  using OpConversionPattern<SPNTask>::OpAdaptor;

  LogicalResult
  matchAndRewrite(SPNTask op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct BatchExtractBufferize : OpConversionPattern<SPNBatchExtract> {

  using OpConversionPattern<SPNBatchExtract>::OpConversionPattern;
  using OpConversionPattern<SPNBatchExtract>::OpAdaptor;

  LogicalResult
  matchAndRewrite(SPNBatchExtract op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

static inline void
populateLoSPNBufferizationPatterns(RewritePatternSet &patterns,
                                   MLIRContext *context,
                                   TypeConverter &typeConverter) {
  patterns.insert<KernelBufferize, TaskBufferize>(typeConverter, context);
  patterns.insert<BatchExtractBufferize>(typeConverter, context);
}

} // namespace low
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_LIB_DIALECT_LOSPN_BUFFERIZE_LOSPNBUFFERIZATIONPATTERNS_H
