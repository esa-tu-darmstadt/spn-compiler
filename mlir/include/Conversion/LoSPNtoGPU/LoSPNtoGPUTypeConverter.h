//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_LOSPNTOGPUTYPECONVERTER_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_LOSPNTOGPUTYPECONVERTER_H

#include <mlir/Transforms/DialectConversion.h>
#include "mlir/IR/BuiltinTypes.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNTypes.h"

namespace mlir {
  namespace spn {

    class LoSPNtoGPUTypeConverter : public TypeConverter {
    public:
      explicit LoSPNtoGPUTypeConverter() {
        addConversion([](FloatType floatType) -> Optional<Type> {
          // FloatType is unconditionally legal.
          return floatType;
        });
        addConversion([](IntegerType intType) -> Optional<Type> {
          // IntegerType is unconditionally legal.
          return intType;
        });
        addConversion([](MemRefType memRefType) -> Optional<Type> {
          // MemRefType are unconditionally legal.
          return memRefType;
        });
        addConversion([](IndexType indexType) -> Optional<Type> {
          // IndexType is unconditionally legal.
          return indexType;
        });
        addConversion([](low::LogType logType) -> Optional<Type> {
          return logType.getBaseType();
        });
        addTargetMaterialization([](OpBuilder& builder, FloatType type,
                                    ValueRange inputs, Location loc) -> Optional<Value> {
          if (inputs.size() != 1) {
            return std::nullopt;
          }
          if (auto logType = inputs[0].getType().dyn_cast<low::LogType>()) {
            if (logType.getBaseType() != type) {
              return std::nullopt;
            }
            return builder.create<low::SPNStripLog>(loc, inputs[0], type).getResult();
          }
          return std::nullopt;
        });
      }
    };

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_LOSPNTOGPUTYPECONVERTER_H
