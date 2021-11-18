//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_UTIL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_UTIL_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
  namespace spn {
    namespace util {
      /*!
       * This utility method for vectors returns following things, depending on the vectors' element bit widths:
       * - an extend operation that extends the input to the target type if the input's bit width is smaller
       * - a truncate operation that truncates the input to the target type if the input's bit width is bigger
       * - the unmodified input if the bit widths match
      */
      Value extendTruncateOrGetVector(Value input, VectorType targetVectorType, RewriterBase& rewriter);

      /*!
       *  This utility method casts vectors to floating point vectors and returns them, depending on their element type:
       *  - create and return an SIToFPOp operation if the input vector contains signed integers
       *  - create and return a UIToFPOp operation if the input vector contains unsigned integers
       *  - return the unmodified input vector if it's a vector with floats already
      */
      Value castToFloatOrGetVector(Value input, VectorType targetFloatVectorType, RewriterBase& rewriter);
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_UTIL_H
