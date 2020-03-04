//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PATTERNS_SPNOPERATIONLOWERING_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PATTERNS_SPNOPERATIONLOWERING_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
  namespace spn {
    template<typename SourceOp>
    class SPNOpLowering : public OpConversionPattern<SourceOp> {

    public:
      SPNOpLowering(MLIRContext* context, TypeConverter& _typeConverter,
                    PatternBenefit benefit = 1) : OpConversionPattern<SourceOp>(context, benefit),
                                                  typeConverter{_typeConverter} {}

    protected:
      TypeConverter& typeConverter;

      LoadOp createStaticLoad(ConversionPatternRewriter& rewriter, Location loc,
                              Value memRef, llvm::ArrayRef<size_t> indices) const;

    };
  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PATTERNS_SPNOPERATIONLOWERING_H
