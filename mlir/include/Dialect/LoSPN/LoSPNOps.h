//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNOPS_H
#define SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/RegionKindInterface.h"
#include "LoSPN/LoSPNInterfaces.h"
#include "LoSPN/LoSPNTraits.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/BuiltinOps.h"

#define GET_OP_CLASSES
#include "LoSPN/LoSPNOps.h.inc"

#endif //SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNOPS_H
