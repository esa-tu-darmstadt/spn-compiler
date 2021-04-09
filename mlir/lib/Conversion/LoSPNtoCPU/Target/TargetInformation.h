//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_DRIVER_TARGET_TARGETINFORMATION_H
#define SPNC_COMPILER_SRC_DRIVER_TARGET_TARGETINFORMATION_H

#include <llvm/ADT/StringMap.h>
#include "mlir/IR/Types.h"

namespace mlir {
  namespace spn {
    class TargetInformation {

    private:

      explicit TargetInformation();

    public:

      TargetInformation(const TargetInformation&) = delete;

      TargetInformation& operator=(const TargetInformation&) = delete;

    public:

      static TargetInformation& nativeCPUTarget();

      bool hasAVX2Support();

      bool hasAVX512Support();

      bool hasAVXSupport();

      unsigned getHWVectorEntries(mlir::Type type);

    private:

      unsigned getHWVectorEntriesAVX512(mlir::Type type);

      unsigned getHWVectorEntriesAVX2(mlir::Type type);

      unsigned getHWVectorEntriesAVX(mlir::Type type);

      llvm::StringMap<bool, llvm::MallocAllocator> featureMap;

    };
  }
}
#endif //SPNC_COMPILER_SRC_DRIVER_TARGET_TARGETINFORMATION_H
