//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "HiSPN/HiSPNDialect.h"
#include "HiSPN/HiSPNOps.h"
#include "HiSPN/HiSPNAttributes.h"
#include "HiSPN/HiSPNInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::spn::high;

void HiSPNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "HiSPN/HiSPNOps.cpp.inc"
  >();
  addTypes<ProbabilityType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "HiSPN/HiSPNAttributes.cpp.inc"
  >();
}

// ::mlir::Type HiSPNDialect::parseType(::mlir::DialectAsmParser& parser) const {
//   return ProbabilityType::get(getContext());
// }

// void HiSPNDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter& os) const {
//   // Currently the only SPN type is the probability type.
//   os << "probability";
// }

// Table-gen output for dialect implementation
#include "HiSPN/HiSPNOpsDialect.cpp.inc"

// Add definitions/implementation of SPN dialect/operation interfaces.
#include "HiSPN/HiSPNInterfaces.cpp.inc"

// Parse a APFloat from an AsmParser. This is required for parsing the value field in HistBucket attributes.
template<> struct FieldParser<APFloat> {
  static FailureOr<APFloat> parse(AsmParser &parser) {
    double value;
    if (parser.parseFloat(value))
      return failure();
    return APFloat(value);
  }
};

#define GET_ATTRDEF_CLASSES
#include "HiSPN/HiSPNAttributes.cpp.inc"