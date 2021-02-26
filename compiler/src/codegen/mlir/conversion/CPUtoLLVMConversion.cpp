//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "CPUtoLLVMConversion.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"

void spnc::CPUtoLLVMConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  pm->nest<mlir::FuncOp>().addPass(mlir::createConvertVectorToSCFPass());
  pm->addPass(mlir::createLowerToCFGPass());
  pm->addPass(mlir::createConvertVectorToLLVMPass());
  pm->addPass(mlir::createLowerToLLVMPass());
}
