//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::spn;

Value util::extendTruncateOrGetVector(Value input, VectorType targetVectorType,
                                      RewriterBase &rewriter) {
  auto inputVectorType = input.getType().dyn_cast<VectorType>();
  assert(inputVectorType &&
         "cannot extend or truncate scalar type to vector type");
  assert(targetVectorType.getElementType().isa<FloatType>() &&
         "target element type must be float type");
  if (inputVectorType.getElementTypeBitWidth() <
      targetVectorType.getElementTypeBitWidth()) {
    return rewriter.create<arith::ExtFOp>(input.getLoc(), targetVectorType,
                                          input);
  } else if (inputVectorType.getElementTypeBitWidth() >
             targetVectorType.getElementTypeBitWidth()) {
    return rewriter.create<arith::TruncFOp>(input.getLoc(), targetVectorType,
                                            input);
  }
  return input;
}

Value util::castToFloatOrGetVector(Value input,
                                   VectorType targetFloatVectorType,
                                   RewriterBase &rewriter) {
  auto inputVectorType = input.getType().dyn_cast<VectorType>();
  assert(inputVectorType && "cannot cast scalar type to float vector type");
  assert(targetFloatVectorType.getElementType().isa<FloatType>() &&
         "target element type must be float type");
  auto inputElementType = inputVectorType.getElementType();
  if (inputElementType.dyn_cast<FloatType>()) {
    return input;
  } else if (auto intElementType = inputElementType.dyn_cast<IntegerType>()) {
    if (intElementType.isSigned()) {
      return rewriter.create<arith::SIToFPOp>(input.getLoc(),
                                              targetFloatVectorType, input);
    }
    return rewriter.create<arith::UIToFPOp>(input.getLoc(),
                                            targetFloatVectorType, input);
  }
  llvm_unreachable("value cannot be cast to float vector");
}
