//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_QUERYPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_QUERYPATTERNS_H

#include "HiSPN/HiSPNDialect.h"
#include "HiSPN/HiSPNOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace spn {

struct JointQueryLowering : OpConversionPattern<high::JointQuery> {

  using OpConversionPattern<high::JointQuery>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(high::JointQuery op, high::JointQuery::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

static inline void
populateHiSPNtoLoSPNQueryPatterns(RewritePatternSet &patterns,
                                  MLIRContext *context,
                                  TypeConverter &typeConverter) {
  patterns.insert<JointQueryLowering>(typeConverter, context);
}

} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_QUERYPATTERNS_H
