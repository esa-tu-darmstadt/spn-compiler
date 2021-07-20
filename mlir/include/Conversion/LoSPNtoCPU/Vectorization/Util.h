//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_UTIL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_UTIL_H

#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"

namespace mlir {
  namespace spn {
    namespace util {

      static inline Value extendTruncateOrGetVector(Value input,
                                                    VectorType targetVectorType,
                                                    RewriterBase& rewriter) {
        auto inputVectorType = input.getType().dyn_cast<VectorType>();
        assert(inputVectorType && "cannot extend or truncate scalar type to vector type");
        assert(targetVectorType.getElementType().isa<FloatType>() && "target element type must be float type");
        if (inputVectorType.getElementTypeBitWidth() < targetVectorType.getElementTypeBitWidth()) {
          return rewriter.create<FPExtOp>(input.getLoc(), input, targetVectorType);
        } else if (inputVectorType.getElementTypeBitWidth() > targetVectorType.getElementTypeBitWidth()) {
          return rewriter.create<FPTruncOp>(input.getLoc(), input, targetVectorType);
        }
        return input;
      }

    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_UTIL_H
