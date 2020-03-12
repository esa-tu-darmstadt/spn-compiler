//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNOpInterfaces.h"

using namespace mlir;

// Compile the operation interface generated via TableGen.
#include "src/codegen/mlir/dialects/spn/SPNOpInterfaces.op.interface.cpp.inc"