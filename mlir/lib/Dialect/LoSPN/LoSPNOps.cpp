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

namespace mlir {
  namespace spn {
    namespace low {

      static mlir::LogicalResult verifyKernel(SPNKernel kernel) {
        // Check that for each input of the SPNKernel the entry block of the Kernel has
        // a block argument with identical type.
        if (kernel.body().front().getNumArguments() != kernel.getType().getNumInputs()) {
          return kernel.emitOpError() << "Number of entry block arguments does not match number of Kernel inputs";
        }
        for (auto inputAndBlockArg : llvm::zip(kernel.getType().getInputs(), kernel.body().front().getArguments())) {
          if (std::get<0>(inputAndBlockArg) != std::get<1>(inputAndBlockArg).getType()) {
            return kernel.emitOpError() << "Kernel input type " << std::get<0>(inputAndBlockArg)
                                        << " and block argument type " << std::get<1>(inputAndBlockArg).getType()
                                        << " did not match";
          }
        }

        // Check that the values returned by each return found directly inside the Kernel
        // match the output type of the Kernel.
        if (!kernel.getType().getResults().empty()) {
          for (auto ret : kernel.body().getOps<SPNReturn>()) {
            if (ret.returnValues().size() != kernel.getType().getNumResults()) {
              return kernel.emitOpError() << "Number of return values does not match Kernel result type";
            }
            for (auto typeAndValue : llvm::zip(kernel.getType().getResults(), ret.returnValues())) {
              auto retTy = std::get<0>(typeAndValue);
              auto valTy = std::get<1>(typeAndValue).getType();
              if (retTy != valTy) {
                return kernel.emitOpError() << "Kernel result type " << retTy
                                            << " did not match return value type " << valTy;
              }
            }
          }
        }
        return mlir::success();
      }

    }
  }
}

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