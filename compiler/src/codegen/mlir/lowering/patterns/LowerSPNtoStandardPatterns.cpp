//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LowerSPNtoStandardPatterns.h"
#include <mlir/IR/Matchers.h>

using namespace mlir;
using namespace mlir::spn;

PatternMatchResult ConstantOpLowering::matchAndRewrite(spn::ConstantOp op, PatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, op.valueAttr());
  return matchSuccess();
}

PatternMatchResult InputVarLowering::matchAndRewrite(InputVarOp op, ArrayRef<Value> operands,
                                                     ConversionPatternRewriter& rewriter) const {
  auto indexAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), op.index().getZExtValue());
  auto indexVal = rewriter.create<mlir::ConstantOp>(op.getLoc(), indexAttr);
  assert(operands.size() == 1 && "Expecting only a single operand!");
  assert(operands[0].getType().isa<MemRefType>() && "Expecting memref as operand!");
  auto load = rewriter.create<mlir::LoadOp>(op.getLoc(), operands[0], ValueRange{indexVal});
  rewriter.replaceOp(op, {load});
  return matchSuccess();
}

PatternMatchResult FunctionLowering::matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                                                     ConversionPatternRewriter& rewriter) const {
  auto fnType = op.getType();

  if (fnType.getNumResults() > 1) {
    return matchFailure();
  }

  auto resType = (fnType.getNumResults()) ? (ArrayRef<Type>) {fnType.getResults()[0]} : llvm::None;

  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  for (auto argType : llvm::enumerate(fnType.getInputs())) {
    auto convertedType = typeConverter.convertType(argType.value());
    if (!convertedType) {
      return matchFailure();
    }
    signatureConverter.addInputs(argType.index(), convertedType);
  }

  auto newFuncOp = rewriter.create<FuncOp>(op.getLoc(), op.getName(),
                                           rewriter.getFunctionType(signatureConverter.getConvertedTypes(), resType),
                                           llvm::None);

  for (const auto& namedAttr : op.getAttrs()) {
    if (!namedAttr.first.is(impl::getTypeAttrName()) &&
        !namedAttr.first.is(SymbolTable::getSymbolAttrName())) {
      newFuncOp.setAttr(namedAttr.first, namedAttr.second);
    }
  }

  rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(), newFuncOp.end());
  rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);
  rewriter.eraseOp(op);
  return matchSuccess();
}
