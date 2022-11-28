//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPN/LoSPNAttributes.h"
#include "LoSPN/LoSPNDialect.h"

namespace mlir::spn::low {

//::mlir::LogicalResult BucketAttr::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int lb, int ub, APFloat val)
//{
//    // TODO:
//    return mlir::success();
//}

}

// required in LoSPNAttributes.cpp.inc
namespace mlir {

template <>
struct FieldParser<llvm::APFloat> {
  static FailureOr<llvm::APFloat> parse(AsmParser &parser) {
    double d;

    if (parser.parseFloat(d))
      return failure();

    return llvm::APFloat(d);
  }
};

}

#define GET_ATTRDEF_CLASSES
#include "LoSPN/LoSPNAttributes.cpp.inc"

namespace mlir::spn::low {
  void LoSPNDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "LoSPN/LoSPNAttributes.cpp.inc"
    >();
  }
}