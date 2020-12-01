//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_INCLUDE_SPN_SPNOPS_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_INCLUDE_SPN_SPNOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/RegionKindInterface.h"
#include "SPN/SPNInterfaces.h"
#include "SPN/SPNEnums.h"

#define GET_OP_CLASSES
#include "SPN/SPNOps.h.inc"

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_INCLUDE_SPN_SPNOPS_H
