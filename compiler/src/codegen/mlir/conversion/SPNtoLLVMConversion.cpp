//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNtoLLVMConversion.h"
#include "SPNtoLLVM/SPNtoLLVMConversionPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

void spnc::SPNtoLLVMConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  //pm->addPass(mlir::spn::createSPNtoLLVMConversionPass());
  //pm->addPass(mlir::createConvertVectorToSCFPass());
  pm->nest<mlir::FuncOp>().addPass(mlir::createConvertVectorToSCFPass());
  pm->addPass(mlir::createLowerToCFGPass());
  pm->addPass(mlir::createConvertVectorToLLVMPass());
}
