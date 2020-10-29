//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOLLVM_SPNTOLLVMPATTERNS_H
#define SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOLLVM_SPNTOLLVMPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/StandardTypes.h"
#include "SPN/SPNOps.h"
#include "SPN/SPNDialect.h"

namespace mlir {
  namespace spn {

    struct HistogramOpLowering : public OpConversionPattern<HistogramOp> {

      using OpConversionPattern<HistogramOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(HistogramOp op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    static void populateSPNtoLLVMConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                    TypeConverter& typeConverter) {
      patterns.insert<HistogramOpLowering>(typeConverter, context);
    }

  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOLLVM_SPNTOLLVMPATTERNS_H
