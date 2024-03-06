//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "CPUtoLLVMConversion.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
// #include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
// #include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void spnc::CPUtoLLVMConversion::initializePassPipeline(mlir::PassManager *pm,
                                                       mlir::MLIRContext *ctx) {
  // pm->addNestedPass<mlir::func::FuncOp>(mlir::createConvertIndexToLLVMPass());
  // pm->addNestedPass<mlir::func::FuncOp>(mlir::createConvertVectorToSCFPass());
  // pm->addNestedPass<mlir::func::FuncOp>(mlir::createConvertSCFToCFPass());
  // pm->addPass(mlir::createConvertVectorToLLVMPass());
  // // pm->addPass(mlir::createArithToLLVMConversionPass());
  // pm->addNestedPass<mlir::func::FuncOp>(mlir::arith::createArithExpandOpsPass());
  // pm->addNestedPass<mlir::func::FuncOp>(mlir::createConvertMathToLLVMPass());
  // pm->addPass(mlir::createConvertFuncToLLVMPass());
  // pm->addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  // // pm->addPass(mlir::createLowerToLLVM());
  // Blanket-convert any remaining high-level vector ops to loops if any remain.
  pm->addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  // Blanket-convert any remaining linalg ops to loops if any remain.
  // pm->addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  // Blanket-convert any remaining affine ops if any remain.
  // pm->addPass(createLowerAffinePass());
  // Convert SCF to CF (always needed).
  pm->addPass(createConvertSCFToCFPass());
  // Sprinkle some cleanups.
  pm->addPass(createCanonicalizerPass());
  pm->addPass(createCSEPass());
  // Blanket-convert any remaining linalg ops to LLVM if any remain.
  // pm->addPass(createConvertLinalgToLLVMPass());
  // Convert vector to LLVM (always needed).
  pm->addPass(createConvertVectorToLLVMPass());
  // Convert Math to LLVM (always needed).
  pm->addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  // Expand complicated MemRef operations before lowering them.
  pm->addPass(memref::createExpandStridedMetadataPass());
  // The expansion may create affine expressions. Get rid of them.
  // pm->addPass(createLowerAffinePass());
  // Convert MemRef to LLVM (always needed).
  pm->addPass(createFinalizeMemRefToLLVMConversionPass());
  // Convert Func to LLVM (always needed).
  pm->addPass(createConvertFuncToLLVMPass());
  // Convert Index to LLVM (always needed).
  pm->addPass(createConvertIndexToLLVMPass());
  // Convert remaining unrealized_casts (always needed).
  pm->addPass(createReconcileUnrealizedCastsPass());
}