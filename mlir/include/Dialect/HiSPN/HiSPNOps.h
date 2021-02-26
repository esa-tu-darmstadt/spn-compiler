//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_DIALECT_HISPN_HISPNOPS_H
#define SPNC_MLIR_INCLUDE_DIALECT_HISPN_HISPNOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/RegionKindInterface.h"
#include "HiSPN/HiSPNInterfaces.h"
#include "HiSPN/HiSPNEnums.h"

#define GET_OP_CLASSES
#include "HiSPN/HiSPNOps.h.inc"

#endif //SPNC_MLIR_INCLUDE_DIALECT_HISPN_HISPNOPS_H
