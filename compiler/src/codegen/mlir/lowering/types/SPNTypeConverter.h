//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_TYPES_SPNTYPECONVERTER_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_TYPES_SPNTYPECONVERTER_H

#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
  namespace spn {
    class SPNTypeConverter : public TypeConverter {

    public:
      explicit SPNTypeConverter() {
        addConversion([](TensorType tensorType) -> Optional<Type> {
          if (!tensorType.hasRank()) {
            return llvm::None;
          }
          return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        });
      }

    };
  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_TYPES_SPNTYPECONVERTER_H
