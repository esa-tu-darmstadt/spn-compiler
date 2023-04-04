//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_LOSPNVECTORIZATIONTYPECONVERTER_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_LOSPNVECTORIZATIONTYPECONVERTER_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
  namespace spn {

    class LoSPNVectorizationTypeConverter : public TypeConverter {
    public:
      explicit LoSPNVectorizationTypeConverter(unsigned vector_width) {
        addConversion([vector_width](FloatType floatType) -> Optional<Type> {
          // All floating point types are converted to vectors of float.
          return VectorType::get(vector_width, floatType);
        });
        addConversion([vector_width](IntegerType intType) -> Optional<Type> {
          // All integer types are converted to vectors of float.
          return VectorType::get(vector_width, intType);
        });
        addConversion([vector_width](low::LogType logType) -> Optional<Type> {
          // The log type is converted to a vector of the base type.
          return VectorType::get(vector_width, logType.getBaseType());
        });
        addConversion([](MemRefType memRefType) -> Optional<Type> {
          // MemRefType are unconditionally legal.
          return memRefType;
        });
        addConversion([](IndexType indexType) -> Optional<Type> {
          // IndexType is unconditionally legal.
          return indexType;
        });
        // from scalar to vector for vectorization.
        addConversion([](VectorType vectorType) -> Optional<Type> {
          // VectorType is unconditionally legal.
          return vectorType;
        });
        addTargetMaterialization([](OpBuilder& builder, VectorType type,
                                    ValueRange inputs, Location loc) -> Optional<Value> {
          if (inputs.size() != 1) {
            return std::nullopt;
          }
          if (auto toScalar = dyn_cast<low::SPNConvertToScalar>(inputs.front().getDefiningOp())) {
            // Handle the special case that the values was previously converted from a vector
            // to a scalar.
            assert(toScalar.getVector().getType() == type);
            return toScalar.getVector();
          }
          return builder.create<low::SPNConvertToVector>(loc, type, inputs.front()).getResult();
        });
      }

    };

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_LOSPNVECTORIZATIONTYPECONVERTER_H
