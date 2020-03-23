//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_TYPES_SPNTYPECONVERTER_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_TYPES_SPNTYPECONVERTER_H

#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
  namespace spn {

    ///
    /// TypeConverter for the SPN dialect, turning Tensors into equivalent MemRefs.
    class SPNTypeConverter : public TypeConverter {

    public:

      ///
      /// Constructor populating the TypeConverter.
      explicit SPNTypeConverter() {
        addConversion([](TensorType tensorType) -> Optional<Type> {
          if (!tensorType.hasRank()) {
            return llvm::None;
          }
          return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        });
        addConversion([](FloatType floatType) -> Optional<Type> {
          if (floatType.getWidth() != 64) {
            return llvm::None;
          }
          return floatType;
        });
        addConversion([](MemRefType memrefType) -> Optional<Type> {
          return memrefType;
        });
        addConversion([](IntegerType intType) -> Optional<Type> {
          if (intType.getWidth() != 32) {
            return llvm::None;
          }
          return intType;
        });
      }

    };
  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_TYPES_SPNTYPECONVERTER_H
