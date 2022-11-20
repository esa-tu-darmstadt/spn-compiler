//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNATTRIBUTES_H
#define SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/AttributeSupport.h"
//#include "mlir/IR/Identifier.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_ATTRDEF_CLASSES
#include "LoSPN/LoSPNAttributes.h.inc"

#endif //SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNATTRIBUTES_H
