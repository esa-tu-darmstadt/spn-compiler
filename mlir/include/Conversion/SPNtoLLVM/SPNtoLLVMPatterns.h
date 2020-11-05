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

    ///
    /// Pattern to lower SPN histogram op directly to LLVM dialect.
    struct HistogramOpLowering : public OpConversionPattern<HistogramOp> {

      using OpConversionPattern<HistogramOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(HistogramOp op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    /// Populate list with all patterns required to lower remaining SPN dialect operations to LLVM dialect.
    /// \param patterns Pattern list to fill.
    /// \param context MLIR context.
    /// \param typeConverter Type converter.
    static void populateSPNtoLLVMConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                    TypeConverter& typeConverter) {
      patterns.insert<HistogramOpLowering>(typeConverter, context);
    }

  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOLLVM_SPNTOLLVMPATTERNS_H
