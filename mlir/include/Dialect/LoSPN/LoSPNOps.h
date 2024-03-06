//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNOPS_H
#define SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNOPS_H

#include "LoSPN/LoSPNAttributes.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNInterfaces.h"
#include "LoSPN/LoSPNTraits.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "LoSPN/LoSPNOps.h.inc"

#endif // SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNOPS_H
