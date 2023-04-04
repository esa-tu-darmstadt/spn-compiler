//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "CPUtoLLVMConversion.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
//#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
//#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"


void spnc::CPUtoLLVMConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  pm->addPass(mlir::createConvertMathToLLVMPass());
  // TODO: not sure how to fix this
  //pm->addPass(mlir::createMemRefToLLVMConversionPass());
  pm->addPass(mlir::createArithToLLVMConversionPass());
  
  pm->nest<mlir::func::FuncOp>().addPass(mlir::createConvertVectorToSCFPass());
  pm->addPass(mlir::createConvertVectorToLLVMPass());
  pm->addPass(mlir::createConvertSCFToCFPass());
  pm->addPass(mlir::createConvertControlFlowToLLVMPass());

  pm->addPass(mlir::createConvertFuncToLLVMPass());

  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createReconcileUnrealizedCastsPass());
}