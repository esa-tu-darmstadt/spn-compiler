//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_CUDA_CUDASERIALIZETOCUBIN_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_CUDA_CUDASERIALIZETOCUBIN_H

#include "mlir/Dialect/GPU/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"

namespace mlir {
  namespace spn {

    ///
    /// MLIR pass to translate the GPUModule inside a MLIR module to CUBIN.
    /// The GPUModule is first translated to NVVM IR, then to PTX assembly via the LLVM PTX backend,
    /// before eventually being translated to CUBIN through calls to the CUDA runtime library.
    class CUDASerializeToCubinPass : public PassWrapper<CUDASerializeToCubinPass, mlir::gpu::SerializeToBlobPass> {

    public:

      explicit CUDASerializeToCubinPass(ArrayRef<llvm::StringRef> kernelFunctions, bool shouldPrintIR = false,
                                        unsigned _optLevel = 2);

    protected:
      void getDependentDialects(DialectRegistry& registry) const override;
    private:
      std::unique_ptr<llvm::Module> translateToLLVMIR(llvm::LLVMContext& llvmContext) override;
      std::unique_ptr<std::vector<char>> serializeISA(const std::string& isa) override;

      std::string getGPUArchitecture();
      void optimizeNVVMIR(llvm::Module* module);
      void linkWithLibdevice(llvm::Module* module);

      bool printIR;

      llvm::SmallVector<llvm::StringRef> kernelFuncs;

      unsigned optLevel;

    };

    std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createSerializeToCubinPass(ArrayRef<StringRef> kernelFunctions,
                                                                                bool shouldPrintIR = false,
                                                                                unsigned optLevel = 2);

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_CUDA_CUDASERIALIZETOCUBIN_H
