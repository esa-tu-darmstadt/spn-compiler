//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDTYPECONVERTER_H
#define SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDTYPECONVERTER_H

#include <mlir/Transforms/DialectConversion.h>
#include "mlir/IR/StandardTypes.h"

namespace mlir {
  namespace spn {

    ///
    /// TypeConverter for the SPN dialects, turning Tensors into equivalent MemRefs.
    class SPNtoStandardTypeConverter : public TypeConverter {

    public:

      ///
      /// Constructor populating the TypeConverter.
      explicit SPNtoStandardTypeConverter() {
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

#endif //SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDTYPECONVERTER_H
