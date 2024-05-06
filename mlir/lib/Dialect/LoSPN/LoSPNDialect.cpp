//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNAttributes.h"
#include "LoSPN/LoSPNInterfaces.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::spn::low;

void LoSPNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LoSPN/LoSPNOps.cpp.inc"
      >();
  addTypes<LogType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "LoSPN/LoSPNAttributes.cpp.inc"
      >();
}

/// Materialize an integer or floating point constant.
Operation *LoSPNDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  return builder.create<SPNConstant>(loc, type, cast<TypedAttr>(value));
  // return mlir::arith::ConstantOp::materialize(builder, value, type, loc);
}

#define GET_TYPEDEF_CLASSES
#include "LoSPN/LoSPNOpsTypes.cpp.inc"

// Table-gen output for dialect implementation
#include "LoSPN/LoSPNOpsDialect.cpp.inc"

// Add definitions/implementation of SPN dialect/operation interfaces.
#include "LoSPN/LoSPNInterfaces.cpp.inc"