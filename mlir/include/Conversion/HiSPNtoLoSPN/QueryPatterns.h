//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_QUERYPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_QUERYPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "HiSPN/HiSPNDialect.h"
#include "HiSPN/HiSPNOps.h"
#include "llvm/Support/Debug.h"

namespace mlir {
  namespace spn {

    struct JointQueryLowering : OpConversionPattern<high::JointQuery> {

      using OpConversionPattern<high::JointQuery>::OpConversionPattern;

      LogicalResult matchAndRewrite(high::JointQuery op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    static void populateHiSPNtoLoSPNQueryPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                  TypeConverter& typeConverter) {
      patterns.insert<JointQueryLowering>(typeConverter, context);
    }

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_QUERYPATTERNS_H
