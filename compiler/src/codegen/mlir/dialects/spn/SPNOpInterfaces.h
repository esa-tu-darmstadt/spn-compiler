//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_DIALECTS_SPN_SPNOPINTERFACES_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_DIALECTS_SPN_SPNOPINTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {

// The interfaces are generated via TableGen, just make them available.
#include "src/codegen/mlir/dialects/spn/SPNOpInterfaces.op.interface.h.inc"

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_DIALECTS_SPN_SPNOPINTERFACES_H
