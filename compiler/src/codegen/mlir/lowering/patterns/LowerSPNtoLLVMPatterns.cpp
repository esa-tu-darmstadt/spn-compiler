//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LowerSPNtoLLVMPatterns.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Module.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>

using namespace mlir;
using namespace spn;

PatternMatchResult HistogramValueLowering::matchAndRewrite(HistogramValueOp op, ArrayRef<Value> operands,
                                                           ConversionPatternRewriter& rewriter) const {
  static int histCount = 0;
  // Construct a global constant array for the histogram values.
  auto llvmDialect = op.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
  auto doubleType = LLVM::LLVMType::getDoubleTy(llvmDialect);
  auto arrType = LLVM::LLVMType::getArrayTy(doubleType, op.values().getNumElements());
  auto module = op.getParentOfType<ModuleOp>();
  auto restore = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(module.getBody());
  auto globalConst = rewriter.create<LLVM::GlobalOp>(op.getLoc(),
                                                     arrType,
                                                     true,
                                                     LLVM::Linkage::Internal,
                                                     "test_" + std::to_string(histCount++),
                                                     op.values());
  rewriter.restoreInsertionPoint(restore);
  // Use the feature input value to index into the global constant array and load the value.
  // First, get the address of the global array, then use GEP to get the address of corresponding to
  // the value at the index given by the input feature value and load the probability at this index.
  auto addressOf = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), globalConst);
  auto indexType = LLVM::LLVMType::getInt64Ty(llvmDialect);
  auto constZeroIndex = rewriter.create<LLVM::ConstantOp>(op.getLoc(), indexType, rewriter.getI64IntegerAttr(0));
  auto ptrType = doubleType.getPointerTo();
  auto gep = rewriter.create<LLVM::GEPOp>(op.getLoc(), ptrType, addressOf, ValueRange{constZeroIndex, constZeroIndex});
  auto llvmType = typeConverter.convertType(op.getType());
  auto memRef = MemRefDescriptor::undef(rewriter, op.getLoc(), llvmType);
  memRef.setAllocatedPtr(rewriter, op.getLoc(), gep);
  memRef.setAlignedPtr(rewriter, op.getLoc(), gep);
  auto const0 = rewriter.create<LLVM::ConstantOp>(op.getLoc(),
                                                  LLVM::LLVMType::getInt64Ty(llvmDialect),
                                                  rewriter.getI64IntegerAttr(0));
  memRef.setOffset(rewriter, op.getLoc(), const0);
  memRef.setConstantSize(rewriter, op.getLoc(), 0, 2);
  memRef.setConstantStride(rewriter, op.getLoc(), 0, 1);
  rewriter.replaceOp(op, {memRef});
  return matchSuccess();
}