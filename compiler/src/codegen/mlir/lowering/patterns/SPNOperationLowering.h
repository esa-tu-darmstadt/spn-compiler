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

    /// Base class for operation conversion patterns for the SPN dialect.
    /// \tparam SourceOp Operation type to be converted.
    template<typename SourceOp>
    class SPNOpLowering : public OpConversionPattern<SourceOp> {

    public:
      /// Constructor.
      /// \param context Surrounding MLIR context.
      /// \param _typeConverter Type converter.
      /// \param benefit Benefit of the conversion, defaults to 1.
      SPNOpLowering(MLIRContext* context, TypeConverter& _typeConverter,
                    PatternBenefit benefit = 1) : OpConversionPattern<SourceOp>(context, benefit),
                                                  typeConverter{_typeConverter} {}

    protected:
      ///
      /// The type converter that should be used for type conversion.
      TypeConverter& typeConverter;

      /// Create a load operation with fixed load indices.
      /// \param rewriter Rewriter used to create and insert operations.
      /// \param loc Location to attach to the generated operations.
      /// \param memRef MemRef to load from.
      /// \param indices List of compile-time constant indices.
      /// \return LoadOp loading the desired value.
      LoadOp createStaticLoad(ConversionPatternRewriter& rewriter, Location loc,
                              Value memRef, llvm::ArrayRef<size_t> indices) const;

    };
  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PATTERNS_SPNOPERATIONLOWERING_H
