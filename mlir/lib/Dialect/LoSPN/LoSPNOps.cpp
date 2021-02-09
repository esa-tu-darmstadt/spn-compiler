//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "LoSPN/LoSPNOps.cpp.inc"