//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUTYPECONVERTER_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUTYPECONVERTER_H

#include <mlir/Transforms/DialectConversion.h>
#include "mlir/IR/BuiltinTypes.h"
#include "LoSPN/LoSPNOps.h"

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
        // TODO Extend for VectorType and add target materialization
        // from scalar to vector for vectorization.
        addConversion([](VectorType vectorType) -> Optional<Type> {
          // VectorType is unconditionally legal.
          return vectorType;
        });
        addTargetMaterialization([](OpBuilder& builder, VectorType type,
                                    ValueRange inputs, Location loc) -> Optional<Value> {
          if (inputs.size() != 1) {
            return llvm::None;
          }
          return builder.create<low::SPNConvertToVector>(loc, type, inputs.front()).getResult();
        });
        addSourceMaterialization([](OpBuilder& builder, FloatType type,
                                    ValueRange inputs, Location loc) -> Optional<Value> {
          llvm::dbgs() << "REQUESTED SOURCE MATERIALIZATION\n";
          inputs.front().dump();
          return llvm::None;
        });
      }
    };

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUTYPECONVERTER_H
