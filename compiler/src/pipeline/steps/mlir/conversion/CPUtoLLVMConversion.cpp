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
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
//#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"


void spnc::CPUtoLLVMConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  /*
  pm->nest<mlir::FuncOp>().addPass(mlir::createConvertVectorToSCFPass());
  pm->addPass(mlir::createLowerToCFGPass());                                        scf -> standard (now math + arith)
  pm->addPass(mlir::createConvertVectorToLLVMPass());                               vector -> llvm
  
  pm->nest<mlir::FuncOp>().addPass(mlir::createConvertMathToLLVMPass());            math -> llvm
  pm->addPass(mlir::createMemRefToLLVMPass());                                      memref -> llvm
  pm->addPass(mlir::createLowerToLLVMPass());                                       standard -> llvm
  */
  
  // TODO: Check if a pass is missing!
  pm->nest<mlir::func::FuncOp>().addPass(mlir::createConvertVectorToSCFPass());
  pm->addPass(mlir::createConvertVectorToLLVMPass());

  // NOTE: Replaced standard -> llvm with arith -> llvm
  pm->nest<mlir::func::FuncOp>().addPass(mlir::createConvertMathToLLVMPass());
  pm->addPass(mlir::createMemRefToLLVMConversionPass());
  pm->addPass(mlir::createArithToLLVMConversionPass());
}