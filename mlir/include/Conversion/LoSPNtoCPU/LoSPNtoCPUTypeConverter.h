//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUTYPECONVERTER_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUTYPECONVERTER_H

#include "LoSPN/LoSPNOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace spn {

class LoSPNtoCPUTypeConverter : public TypeConverter {
public:
  explicit LoSPNtoCPUTypeConverter(bool convertLog = true) {
    addConversion([](FloatType floatType) -> std::optional<Type> {
      // FloatType is unconditionally legal.
      return floatType;
    });
    addConversion([](IntegerType intType) -> std::optional<Type> {
      // IntegerType is unconditionally legal.
      return intType;
    });
    addConversion([](MemRefType memRefType) -> std::optional<Type> {
      // MemRefType are unconditionally legal.
      return memRefType;
    });
    addConversion([](IndexType indexType) -> std::optional<Type> {
      // IndexType is unconditionally legal.
      return indexType;
    });
    addConversion([convertLog](low::LogType logType) -> std::optional<Type> {
      if (convertLog)
        return logType.getBaseType();
      else
        return logType;
    });
    addTargetMaterialization([](OpBuilder &builder, FloatType type,
                                ValueRange inputs,
                                Location loc) -> std::optional<Value> {
      if (inputs.size() != 1) {
        return std::nullopt;
      }
      if (auto logType = inputs[0].getType().dyn_cast<low::LogType>()) {
        if (logType.getBaseType() != type) {
          return std::nullopt;
        }
        return builder.create<low::SPNStripLog>(loc, inputs[0], type)
            .getResult();
      }
      return std::nullopt;
    });
  }
};

} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUTYPECONVERTER_H
