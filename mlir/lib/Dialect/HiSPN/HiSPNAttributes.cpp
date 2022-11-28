//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "HiSPN/HiSPNAttributes.h"
#include "HiSPN/HiSPNDialect.h"

// required in HiSPNAttributes.cpp.inc
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
#include "HiSPN/HiSPNAttributes.cpp.inc"

namespace mlir::spn::high {
  void HiSPNDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "HiSPN/HiSPNAttributes.cpp.inc"
    >();
  }
}



//#define GET_ATTRDEF_CLASSES
//#include "HiSPN/HiSPNAttributes.cpp.inc"
