//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNAttributes.h"
#include "LoSPN/LoSPNInterfaces.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::spn::low;

void LoSPNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LoSPN/LoSPNOps.cpp.inc"
  >();
  addTypes<LogType>();
}

::mlir::Type LoSPNDialect::parseType(::mlir::DialectAsmParser& parser) const {
  // Currently only handles the LogType, as no other dialect types are defined for LoSPN.

  // LogType is printed as "log<$baseType>"
  if (parser.parseKeyword("log") || parser.parseLess()) {
    // Failed to parse "log" or "<"
    return Type();
  }
  mlir::Type baseType;
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseType(baseType)) {
    // Failed to parse the base type.
    return Type();
  }
  if (!baseType.isa<FloatType>()) {
    parser.emitError(typeLoc, "Base type must be a float type");
    return Type();
  }

  if (parser.parseGreater()) {
    // Failed to parse closing ">"
    return Type();
  }

  return LogType::get(baseType);
}

void LoSPNDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter& os) const {
  // Currently only handles the LogType, as no other dialect types are defined for LoSPN.
  LogType logType = type.cast<LogType>();

  // LogType is printed as "log<$baseType>"
  os << "log<" << logType.getBaseType() << ">";
}

// Table-gen output for dialect implementation
#include "LoSPN/LoSPNOpsDialect.cpp.inc"

// Add definitions/implementation of SPN dialect/operation interfaces.
#include "LoSPN/LoSPNInterfaces.cpp.inc"