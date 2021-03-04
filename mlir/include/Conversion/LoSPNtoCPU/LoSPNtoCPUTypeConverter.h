//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUTYPECONVERTER_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUTYPECONVERTER_H

#include <mlir/Transforms/DialectConversion.h>
#include "mlir/IR/BuiltinTypes.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNTypes.h"

namespace mlir {
  namespace spn {

    class LoSPNtoCPUTypeConverter : public TypeConverter {
    public:
      explicit LoSPNtoCPUTypeConverter() {
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
            return llvm::None;
          }
          if (auto logType = inputs[0].getType().dyn_cast<low::LogType>()) {
            if (logType.getBaseType() != type) {
              return llvm::None;
            }
            return builder.create<low::SPNStripLog>(loc, inputs[0], type).getResult();
          }
          return llvm::None;
        });
      }
    };

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUTYPECONVERTER_H
