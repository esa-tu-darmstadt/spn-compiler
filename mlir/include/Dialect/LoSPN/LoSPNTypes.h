//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNTYPES_H
#define SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNTYPES_H

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir {
  namespace spn {
    namespace low {

      struct LogTypeStorage : public TypeStorage {

        explicit LogTypeStorage(Type _baseType) : baseType(_baseType) {}

        using KeyTy = Type;

        bool operator==(const KeyTy& key) const {
          return key == baseType;
        }

        static LogTypeStorage* construct(TypeStorageAllocator& allocator,
                                         const KeyTy& key) {
          return new(allocator.allocate<LogTypeStorage>()) LogTypeStorage(key);
        }

        Type baseType;

      };

      class LogType : public Type::TypeBase<LogType, Type, LogTypeStorage> {

      public:

        using Base::Base;

        static LogType get(Type baseType) {
          return Base::get(baseType.getContext(), baseType);
        }

        static LogType getChecked(Location loc, Type baseType) {
          return Base::getChecked(loc, baseType);
        }

        static LogicalResult verifyConstructionInvariants(Location loc, Type baseType) {
          if (!baseType.isa<FloatType>()) {
            return emitError(loc) << "Log-space computation currently supported via floating-point types";
          }
          return success();
        }

        Type getBaseType() {
          return getImpl()->baseType;
        }

      };

    }
  }
}

#endif //SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNTYPES_H
