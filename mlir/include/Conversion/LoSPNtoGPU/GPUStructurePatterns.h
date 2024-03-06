//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_STRUCTUREPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_STRUCTUREPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "llvm/Support/Debug.h"

namespace mlir {
  namespace spn {

    struct KernelGPULowering : OpConversionPattern<low::SPNKernel> {

      using OpConversionPattern<low::SPNKernel>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNKernel op,
                                    low::SPNKernel::Adaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct BatchTaskGPULowering : OpConversionPattern<low::SPNTask> {

      using OpConversionPattern<low::SPNTask>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNTask op,
                                    low::SPNTask::Adaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct BodyGPULowering : OpConversionPattern<low::SPNBody> {

      using OpConversionPattern<low::SPNBody>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBody op,
                                    low::SPNBody::Adaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    static inline void populateLoSPNtoGPUStructurePatterns(RewritePatternSet& patterns, MLIRContext* context,
                                                    TypeConverter& typeConverter) {
      patterns.insert<KernelGPULowering, BatchTaskGPULowering>(typeConverter, context);
      patterns.insert<BodyGPULowering>(typeConverter, context);
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_STRUCTUREPATTERNS_H
