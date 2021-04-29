//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

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
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct BatchTaskGPULowering : OpConversionPattern<low::SPNTask> {

      using OpConversionPattern<low::SPNTask>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNTask op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct BodyGPULowering : OpConversionPattern<low::SPNBody> {

      using OpConversionPattern<low::SPNBody>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBody op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    static inline void populateLoSPNtoGPUStructurePatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                    TypeConverter& typeConverter) {
      patterns.insert<KernelGPULowering, BatchTaskGPULowering>(typeConverter, context);
      patterns.insert<BodyGPULowering>(typeConverter, context);
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_STRUCTUREPATTERNS_H
