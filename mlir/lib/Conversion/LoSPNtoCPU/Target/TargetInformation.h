//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

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

      bool hasNeonSupport();

      unsigned getHWVectorEntries(mlir::Type type);

    private:

      unsigned getHWVectorEntriesAVX512(mlir::Type type);

      unsigned getHWVectorEntriesAVX2(mlir::Type type);

      unsigned getHWVectorEntriesAVX(mlir::Type type);

      unsigned getHWVectorEntriesNeon(mlir::Type type);

      llvm::StringMap<bool, llvm::MallocAllocator> featureMap;

    };
  }
}
#endif //SPNC_COMPILER_SRC_DRIVER_TARGET_TARGETINFORMATION_H
