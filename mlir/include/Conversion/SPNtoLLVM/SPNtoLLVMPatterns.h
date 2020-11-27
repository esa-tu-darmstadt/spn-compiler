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
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

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

    ///
    /// Pattern to lower SPN Categorical leaf directly to LLVM dialect.
    struct CategoricalOpLowering : public OpConversionPattern<CategoricalOp> {

      using OpConversionPattern<CategoricalOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(CategoricalOp op,
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
      patterns.insert<CategoricalOpLowering>(typeConverter, context);
    }

    template<typename SourceOp>
    mlir::LogicalResult replaceOpWithGlobalArrayLoad(SourceOp op, ConversionPatternRewriter& rewriter,
                                                     TypeConverter& typeConverter, Value indexOperand,
                                                     ArrayRef<Attribute> arrayValues) {
      static int tableCount = 0;
      auto resultType = op.getResult().getType();
      if (!resultType.isIntOrFloat()) {
        // Currently only handling Int and Float result types.
        return failure();
      }

      // The MLIR to LLVM bridge can only handle ElementsAttr for arrays, so construct one here
      auto rankedType = RankedTensorType::get({(long) arrayValues.size()}, resultType);
      auto valArrayAttr = DenseElementsAttr::get(rankedType, arrayValues);

      // Create & insert a constant global array with the values from the histogram.
      auto elementType = typeConverter.convertType(resultType).template dyn_cast<mlir::LLVM::LLVMType>();
      assert(elementType);
      auto arrType = LLVM::LLVMType::getArrayTy(elementType, arrayValues.size());
      auto module = op.template getParentOfType<ModuleOp>();
      auto restore = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(module.getBody());
      auto globalConst = rewriter.create<LLVM::GlobalOp>(op.getLoc(),
                                                         arrType,
                                                         true,
                                                         LLVM::Linkage::Internal,
                                                         "table_" + std::to_string(tableCount++),
                                                         valArrayAttr);
      rewriter.restoreInsertionPoint(restore);

      // Load a value from the histogram using the index value to index into the histogram.
      auto addressOf = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), globalConst);
      auto indexType = LLVM::LLVMType::getInt64Ty(rewriter.getContext());
      auto constZeroIndex = rewriter.create<LLVM::ConstantOp>(op.getLoc(), indexType, rewriter.getI64IntegerAttr(0));
      auto ptrType = elementType.getPointerTo();
      auto
          gep = rewriter.create<LLVM::GEPOp>(op.getLoc(), ptrType, addressOf, ValueRange{constZeroIndex, indexOperand});
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, gep);
      return success();
    }

  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOLLVM_SPNTOLLVMPATTERNS_H
