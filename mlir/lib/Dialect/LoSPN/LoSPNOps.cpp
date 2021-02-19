//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNAttributes.h"
#include "LoSPN/LoSPNInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

//===----------------------------------------------------------------------===//
// SPNKernel
//===----------------------------------------------------------------------===//

void mlir::spn::low::SPNKernel::build(::mlir::OpBuilder& odsBuilder,
                                      ::mlir::OperationState& odsState,
                                      StringRef name,
                                      FunctionType type) {
  auto nameAttr = odsBuilder.getStringAttr(name);
  auto typeAttr = TypeAttr::get(type);
  odsState.addAttribute(SymbolTable::getSymbolAttrName(), nameAttr);
  odsState.addAttribute(getTypeAttrName(), typeAttr);
  odsState.addAttribute("kernelType", typeAttr);
  odsState.addAttribute("kernelName", nameAttr);
  odsState.addRegion();
}

//===----------------------------------------------------------------------===//
// SPNConstant
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::spn::low::SPNConstant::fold(::llvm::ArrayRef<::mlir::Attribute> operands) {
  assert(operands.empty() && "lo_spn.constant has no operands");
  return valueAttr();
}

//===----------------------------------------------------------------------===//
// SPNGaussianLeaf
//===----------------------------------------------------------------------===//

bool mlir::spn::low::SPNGaussianLeaf::isVectorizable(unsigned vectorFactor) {
  // Floating point narrowing (FPTrunc) and widening (FPExt) cannot be performed in
  // vectorized mode, hence vectorization is not possible if such a transformation
  // of the input value is required.
  if (auto inputFloatType = this->index().getType().dyn_cast<FloatType>()) {
    if (auto outputFloatType = this->getResult().getType().dyn_cast<FloatType>()) {
      if (inputFloatType.getWidth() != outputFloatType.getWidth()) {
        return false;
      }
    }
  }
  return true;
}

#define GET_OP_CLASSES
#include "LoSPN/LoSPNOps.cpp.inc"