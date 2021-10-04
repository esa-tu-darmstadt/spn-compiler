//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "GPUtoLLVMConversion.h"
#include "CUDA/CUDASerializeToCubin.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/ADT/SmallSet.h"
#include <driver/GlobalOptions.h>

mlir::ModuleOp& spnc::GPUtoLLVMConversion::execute() {
  if (!cached) {
    // Initialize LLVM NVPTX backend, as we will lower the
    // content of the GPU module to PTX and compile it to cubin.
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
    //
    // The lowering of the GPU-part of the input module involves the
    // following steps:
    // 1. Outline the GPU parts from the host part of the module.
    // 2. Run a bunch of transformation passes on the GPU-portion of the module.
    // 3. Convert the GPU kernels to a binary blob. For this purpose, the
    //    GPU portion of the module is translated to NVVM IR (essentially LLVM IR with some extensions)
    //    and compiled to PTX assembly using LLVM's PTX backend. The generated PTX is then compiled and
    //    linked into CUBIN using the CUDA runtime library API.
    //    The binary representation of the CUBIN is attached to the MLIR module as a
    //    string attribute and will be included as binary blob in the compiler output.
    //    At runtime, the binary blob is loaded with the CUDA API and executed.
    // 4. Apply the conversion to async calls to all the GPU calls on the host side.
    // 5. Replace the calls to GPU management functions on the host side with calls
    //    to a very thin runtime wrapper library around the CUDA API. This step also
    //    lowers the remaining code from standard to LLVM dialect.
    // 6. Lower the newly generated calls to LLVM dialect.
    // The result of this transformation is a MLIR module with only the host-part remaining as MLIR
    // code (the GPU portion is a binary blob attribute) in LLVM dialect that can then be lowered to LLVM IR.
    // Enable IR printing if requested via CLI

    // Setup the pass manager.
    mlir::PassManager pm{mlirContext.get()};
    if (spnc::option::dumpIR.get(*this->config)) {
      pm.enableIRPrinting(/* Print before every pass*/ [](mlir::Pass*, mlir::Operation*) { return false; },
          /* Print after every pass*/ [](mlir::Pass*, mlir::Operation*) { return true; },
          /* Print module scope*/ true,
          /* Print only after change*/ false);
    }
    auto inputModule = input.execute();
    // Clone the module to keep the original module available
    // for actions using the same input module.
    module = std::make_unique<mlir::ModuleOp>(inputModule.clone());

    // Collect all the GPU kernel function names present in the module
    // to preserve them during internalization.
    // This list is passed to translateAndLinkGPUModule.
    // NOTE: The GPU kernels should already have been outlined by
    // the GPUKernelOutliningPass.
    llvm::SmallVector<llvm::StringRef, 5> gpuKernels;
    module->walk([&gpuKernels](mlir::gpu::GPUFuncOp op) {
      gpuKernels.push_back(op.getName());
    });
    // Retrieve option value to pass to translateAndLinkGPUModule.
    auto printIR = spnc::option::dumpIR.get(*config);

    // Lower SCF constructs to CFG structure.
    pm.addPass(mlir::createLowerToCFGPass());
    // Nested pass manager operating only on the GPU-part of the code.
    auto& kernelPm = pm.nest<mlir::gpu::GPUModuleOp>();
    kernelPm.addPass(mlir::createStripDebugInfoPass());
    kernelPm.addPass(mlir::createLowerGpuOpsToNVVMOpsPass());
    // Convert the GPU-part to a binary blob and annotate it as an atttribute to the MLIR module.
    kernelPm.addPass(mlir::spn::createSerializeToCubinPass(gpuKernels, printIR, irOptLevel));
    auto& funcPm = pm.nest<mlir::FuncOp>();
    funcPm.addPass(mlir::createStdExpandOpsPass());
    funcPm.addPass(mlir::createGpuAsyncRegionPass());
    funcPm.addPass(mlir::createAsyncRuntimeRefCountingPass());
    // Convert the host-side GPU operations into runtime library calls.
    // This also lowers Standard-dialect operations to LLVM dialect.
    pm.addPass(mlir::createGpuToLLVMConversionPass());
    pm.addPass(mlir::createAsyncToAsyncRuntimePass());
    pm.addPass(mlir::createConvertAsyncToLLVMPass());
    pm.addPass(mlir::createLowerToLLVMPass());

    auto result = pm.run(*module);
    if (failed(result)) {
      SPNC_FATAL_ERROR("Converting the GPU module failed");
    }
    auto verificationResult = module->verify();
    if (failed(verificationResult)) {
      SPNC_FATAL_ERROR("Module failed verification after conversion of GPU code");
    }
    cached = true;
  }
  return *module;
}

spnc::GPUtoLLVMConversion::GPUtoLLVMConversion(ActionWithOutput<mlir::ModuleOp>& input,
                                               std::shared_ptr<mlir::MLIRContext> ctx,
                                               unsigned optLevel) :
    ActionSingleInput<mlir::ModuleOp, mlir::ModuleOp>(input),
    mlirContext{std::move(ctx)},
    irOptLevel{optLevel} {}