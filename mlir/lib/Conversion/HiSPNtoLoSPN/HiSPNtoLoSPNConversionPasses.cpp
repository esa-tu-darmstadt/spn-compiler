//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h"
#include "HiSPN/HiSPNOps.h"
#include "HiSPNtoLoSPN/ArithmeticPrecisionAnalysis.h"
#include "HiSPNtoLoSPN/HiSPNTypeConverter.h"
#include "HiSPNtoLoSPN/NodePatterns.h"
#include "LoSPN/LoSPNDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include <HiSPNtoLoSPN/QueryPatterns.h>

using namespace mlir;
namespace {
FailureOr<FloatType> getComputeType(mlir::MLIRContext *context, int width) {
  switch (width) {
  case 32:
    return FloatType::getF32(context);
  case 64:
    return FloatType::getF64(context);
  default:
    return failure();
  }
}
} // namespace

namespace mlir::spn {

#define GEN_PASS_DEF_HISPNTOLOSPNNODECONVERSIONPASS
#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h.inc"
struct HiSPNtoLoSPNNodeConversionPass
    : public mlir::spn::impl::HiSPNtoLoSPNNodeConversionPassBase<
          HiSPNtoLoSPNNodeConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<high::HiSPNDialect>();
    target.addLegalDialect<low::LoSPNDialect>();
    target.addLegalOp<ModuleOp>();

    target.addIllegalOp<high::ProductNode, high::SumNode, high::HistogramNode,
                        high::CategoricalNode, high::GaussianNode,
                        high::RootNode>();

    // Use type analysis to determine data type for actual computation.
    // The concrete type determined by the analysis replaces the abstract
    // probability type used by the HiSPN dialect.
    HiSPNTypeConverter typeConverter;
    if (optimizeRepresentation) {
      auto &arithmeticAnalysis =
          getAnalysis<mlir::spn::ArithmeticPrecisionAnalysis>();
      typeConverter = HiSPNTypeConverter(
          arithmeticAnalysis.getComputationType(computeLogSpace));
    } else {
      int effectiveWidth =
          computeLogSpace ? logComputeTypeWidth : computeTypeWidth;
      FailureOr<FloatType> maybeComputeType =
          getComputeType(&getContext(), effectiveWidth);

      if (failed(maybeComputeType)) {
        mlir::emitError(getOperation().getLoc(),
                        "Unsupported floating-point type width: ")
            << effectiveWidth;
        signalPassFailure();
      }
      typeConverter = HiSPNTypeConverter(maybeComputeType.value());
    }

    RewritePatternSet patterns(&getContext());
    mlir::spn::populateHiSPNtoLoSPNNodePatterns(patterns, &getContext(),
                                                typeConverter);

    auto op = getOperation();
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(op, target, frozenPatterns))) {
      signalPassFailure();
    }
    // Explicitly mark the ArithmeticPrecisionAnalysis as preserved, so the
    // QueryConversionPass can use the information, even though the Graph's
    // nodes have already been converted.
    markAnalysesPreserved<ArithmeticPrecisionAnalysis>();
  }
};

#define GEN_PASS_DEF_HISPNTOLOSPNQUERYCONVERSIONPASS
#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h.inc"
struct HiSPNtoLoSPNQueryConversionPass
    : public mlir::spn::impl::HiSPNtoLoSPNQueryConversionPassBase<
          HiSPNtoLoSPNQueryConversionPass> {
  using Base::Base;
  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<low::LoSPNDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<func::FuncOp>();

    target.addIllegalDialect<high::HiSPNDialect>();

    // Use type analysis to determine data type for actual computation.
    // The concrete type determined by the analysis replaces the abstract
    // probability type used by the HiSPN dialect.
    HiSPNTypeConverter typeConverter;
    if (optimizeRepresentation) {
      auto &arithmeticAnalysis =
          getAnalysis<mlir::spn::ArithmeticPrecisionAnalysis>();
      typeConverter = HiSPNTypeConverter(
          arithmeticAnalysis.getComputationType(computeLogSpace));
    } else {
      int effectiveWidth =
          computeLogSpace ? logComputeTypeWidth : computeTypeWidth;
      FailureOr<FloatType> maybeComputeType =
          getComputeType(&getContext(), effectiveWidth);

      if (failed(maybeComputeType)) {
        mlir::emitError(getOperation().getLoc(),
                        "Unsupported floating-point type width: ")
            << effectiveWidth;
        signalPassFailure();
      }
      typeConverter = HiSPNTypeConverter(maybeComputeType.value());
    }

    RewritePatternSet patterns(&getContext());
    mlir::spn::populateHiSPNtoLoSPNQueryPatterns(patterns, &getContext(),
                                                 typeConverter);

    auto op = getOperation();
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyFullConversion(op, target, frozenPatterns))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir::spn