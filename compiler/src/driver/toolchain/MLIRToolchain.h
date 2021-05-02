//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_MLIRTOOLCHAIN_H
#define SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_MLIRTOOLCHAIN_H

#include "mlir/IR/BuiltinOps.h"
#include <driver/Job.h>
#include <driver/Options.h>
#include <llvm/Target/TargetMachine.h>

namespace spnc {

  ///
  /// Common functionality for all tool-chains.
  class MLIRToolchain {

  protected:

    static void initializeMLIRContext(mlir::MLIRContext& ctx);

    static std::shared_ptr<mlir::ScopedDiagnosticHandler> setupDiagnosticHandler(mlir::MLIRContext* ctx);

    static std::shared_ptr<llvm::TargetMachine> createTargetMachine(bool cpuVectorize);

    static llvm::SmallVector<std::string> parseLibrarySearchPaths(const std::string& paths);

  };

}

#endif //SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_MLIRTOOLCHAIN_H