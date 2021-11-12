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
#include "option/Options.h"
#include "Kernel.h"
#include "llvm/Target/TargetMachine.h"

namespace spnc {

  ///
  /// Information about libraries that should be linked to the executable.
  // Comprises names of the libraries and potential search paths.
  class LibraryInfo {

  public:

    LibraryInfo(llvm::ArrayRef<std::string> libraries, llvm::ArrayRef<std::string> searchPaths) :
        libs(libraries.begin(), libraries.end()), paths(searchPaths.begin(), searchPaths.end()) {}

    llvm::ArrayRef<std::string> libraries() {
      return libs;
    }

    llvm::ArrayRef<std::string> searchPaths() {
      return paths;
    }

  private:

    llvm::SmallVector<std::string> libs;

    llvm::SmallVector<std::string> paths;

  };

  ///
  /// Simple struct to carry information about the generated kernel between different
  /// steps of the tool-chain.
  struct KernelInfo {
    spnc::KernelQueryType queryType;
    spnc::KernelTarget target;
    unsigned batchSize;
    unsigned numFeatures;
    unsigned bytesPerFeature;
    unsigned numResults;
    unsigned bytesPerResult;
    std::string dtype;
    std::string kernelName;
  };

  ///
  /// Common functionality for all tool-chains.
  class MLIRToolchain {

  protected:

    static void initializeMLIRContext(mlir::MLIRContext& ctx);

    static std::unique_ptr<mlir::ScopedDiagnosticHandler> setupDiagnosticHandler(mlir::MLIRContext* ctx);

    static std::unique_ptr<llvm::TargetMachine> createTargetMachine(int optLevel);

    static llvm::SmallVector<std::string> parseLibrarySearchPaths(const std::string& paths);

  };

}

#endif //SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_MLIRTOOLCHAIN_H