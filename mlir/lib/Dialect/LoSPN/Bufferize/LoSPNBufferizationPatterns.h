//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_LIB_DIALECT_LOSPN_BUFFERIZE_LOSPNBUFFERIZATIONPATTERNS_H
#define SPNC_MLIR_LIB_DIALECT_LOSPN_BUFFERIZE_LOSPNBUFFERIZATIONPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "llvm/Support/Debug.h"

namespace mlir {
  namespace spn {
    namespace low {

      struct KernelBufferize : OpConversionPattern<SPNKernel> {

        using OpConversionPattern<SPNKernel>::OpConversionPattern;

        LogicalResult matchAndRewrite(SPNKernel op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter& rewriter) const override;
      };

      struct TaskBufferize : OpConversionPattern<SPNTask> {

        using OpConversionPattern<SPNTask>::OpConversionPattern;

        LogicalResult matchAndRewrite(SPNTask op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter& rewriter) const override;

      };

      struct BatchExtractBufferize : OpConversionPattern<SPNBatchExtract> {

        using OpConversionPattern<SPNBatchExtract>::OpConversionPattern;

        LogicalResult matchAndRewrite(SPNBatchExtract op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter& rewriter) const override;
      };

      static inline void populateLoSPNBufferizationPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                     TypeConverter& typeConverter) {
        patterns.insert<KernelBufferize, TaskBufferize>(typeConverter, context);
        patterns.insert<BatchExtractBufferize>(typeConverter, context);
      }

    }
  }
}

#endif //SPNC_MLIR_LIB_DIALECT_LOSPN_BUFFERIZE_LOSPNBUFFERIZATIONPATTERNS_H
