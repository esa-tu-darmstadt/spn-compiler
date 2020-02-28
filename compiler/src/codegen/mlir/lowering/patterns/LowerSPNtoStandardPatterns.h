//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PATTERNS_LOWERSPNTOSTANDARDPATTERNS_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PATTERNS_LOWERSPNTOSTANDARDPATTERNS_H

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include <codegen/mlir/dialects/spn/SPNDialect.h>
#include <codegen/mlir/lowering/types/SPNTypeConverter.h>

namespace mlir {
  namespace spn {

    struct ConstantOpLowering : public OpRewritePattern<spn::ConstantOp> {

      using OpRewritePattern<spn::ConstantOp>::OpRewritePattern;

      PatternMatchResult matchAndRewrite(spn::ConstantOp op, PatternRewriter& rewriter) const final;

    };

    template<typename SourceOp>
    class SPNOpLowering : public OpConversionPattern<SourceOp> {

    public:
      SPNOpLowering(MLIRContext* context, SPNTypeConverter& _typeConverter,
                    PatternBenefit benefit = 1) : OpConversionPattern<SourceOp>(context, benefit),
                                                  typeConverter{_typeConverter} {}

    protected:
      SPNTypeConverter& typeConverter;
    };

    struct InputVarLowering : public SPNOpLowering<InputVarOp> {
      using SPNOpLowering<InputVarOp>::SPNOpLowering;

      PatternMatchResult matchAndRewrite(InputVarOp op, ArrayRef<Value> operands,
                                         ConversionPatternRewriter& rewriter) const override;
    };

    struct FunctionLowering : public SPNOpLowering<FuncOp> {
      using SPNOpLowering<FuncOp>::SPNOpLowering;

      PatternMatchResult matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                                         ConversionPatternRewriter& rewriter) const override;

    };

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PATTERNS_LOWERSPNTOSTANDARDPATTERNS_H
