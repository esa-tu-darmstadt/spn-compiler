//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_LIB_DIALECT_LOSPN_BUFFERIZE_LOSPNBUFFERIZATIONPATTERNS_H
#define SPNC_MLIR_LIB_DIALECT_LOSPN_BUFFERIZE_LOSPNBUFFERIZATIONPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "llvm/Support/Debug.h"

namespace mlir {
  namespace spn {
    namespace low {

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

      static void populateLoSPNBufferizationPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                     TypeConverter& typeConverter) {
        patterns.insert<TaskBufferize>(typeConverter, context);
        patterns.insert<BatchExtractBufferize>(typeConverter, context);
      }

    }
  }
}

#endif //SPNC_MLIR_LIB_DIALECT_LOSPN_BUFFERIZE_LOSPNBUFFERIZATIONPATTERNS_H
