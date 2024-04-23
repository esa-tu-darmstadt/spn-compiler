//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "CUDASerializeToCubin.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "util/Logging.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Coroutines.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <cuda.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/TargetRegistry.h>

#ifndef SPNC_LIBDEVICE_FILE
// The location of libdevice is usually auto-detected and set by CMake.
#define SPNC_LIBDEVICE_FILE "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"
#endif

inline void emit_cuda_error(const llvm::Twine &message, const char *buffer,
                            CUresult error) {
  SPNC_FATAL_ERROR(message.concat(" failed with error code ")
                       .concat(llvm::Twine{error})
                       .concat("[")
                       .concat(buffer)
                       .concat("]")
                       .str());
}

#define RETURN_ON_CUDA_ERROR(expr, msg)                                        \
  {                                                                            \
    auto _cuda_error = (expr);                                                 \
    if (_cuda_error != CUDA_SUCCESS) {                                         \
      emit_cuda_error(msg, jitErrorBuffer, _cuda_error);                       \
      return {};                                                               \
    }                                                                          \
  }

void mlir::spn::CUDASerializeToCubinPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  registerNVVMDialectTranslation(registry);
  SerializeToBlobPass::getDependentDialects(registry);
}

std::unique_ptr<llvm::Module>
mlir::spn::CUDASerializeToCubinPass::translateToLLVMIR(
    llvm::LLVMContext &llvmContext) {
  // Apply fast-math flags to all floating-point operations and functions in the
  // GPU module.
  // TODO Reason about reassoc flag.
  auto fmf = mlir::LLVM::FastmathFlags::nsz |
             mlir::LLVM::FastmathFlags::contract |
             mlir::LLVM::FastmathFlags::arcp | mlir::LLVM::FastmathFlags::afn;
  auto gpuModule = getOperation();
  gpuModule->walk([fmf](mlir::Operation *op) {
    if (auto fmi = mlir::dyn_cast<mlir::LLVM::FastmathFlagsInterface>(op)) {
      fmi->setAttr("fastmathFlags",
                   mlir::LLVM::FMFAttr::get(fmi->getContext(), fmf));
    }
  });
  // Translate the input MLIR GPU module to NVVM IR (LLVM IR + some extension).
  auto llvmModule = mlir::translateModuleToLLVMIR(gpuModule, llvmContext,
                                                  "LLVMDialectModule");
  if (!llvmModule) {
    SPNC_FATAL_ERROR("Translation of GPU code to NVVM IR failed");
  }

  // Link the generated LLVM/NVVM IR with libdevice.
  linkWithLibdevice(llvmModule.get());

  if (optLevel > 0) {
    // Apply optimization passes to the LLVM/NVVM IR after linking with
    // libdevice.
    optimizeNVVMIR(llvmModule.get());
  }

  if (this->printIR) {
    llvm::dbgs() << "// *** IR Dump After conversion and optimization of NVVM "
                    "IR ***\n\n";
    llvmModule->dump();
    llvm::dbgs() << "\n";
  }
  return llvmModule;
}

void mlir::spn::CUDASerializeToCubinPass::optimizeNVVMIR(llvm::Module *module) {
  // Set the nvvm-reflect-ftz flag to enable/disable use of fast-paths flushing
  // subnormals to zero during the NVVMReflect pass.
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz", true);

  // Setup target machine.
  std::string errorMessage;
  auto target =
      llvm::TargetRegistry::lookupTarget("nvptx64-nvidia-cuda", errorMessage);
  if (!target) {
    SPNC_FATAL_ERROR("Failed to get target for NVPTX: {}", errorMessage);
  }
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      this->triple, this->chip, this->features, {}, {}));

  // Create and populate pass manager.
  // This is copy & pasta code from OptUtils.cpp, with the important difference
  // being that we let the target machine adjust the pass manager, adding the
  // NVVMReflectPass to our pass pipeline.
  llvm::legacy::PassManager modulePM;
  llvm::legacy::FunctionPassManager funcPM(module);
  llvm::PassManagerBuilder builder;
  builder.OptLevel = optLevel;
  builder.SizeLevel = 0;
  builder.Inliner = llvm::createFunctionInliningPass(
      optLevel, 0, /*DisableInlineHotCallSite=*/false);
  builder.LoopVectorize = false;      // Not required on GPU
  builder.SLPVectorize = false;       // Not required on GPU
  builder.DisableUnrollLoops = false; // Allow loop unrolling.

  // Add all coroutine passes to the builder.
  llvm::addCoroutinePassesToExtensionPoints(builder);

  if (machine) {
    // Adjust the pass manager, which adds the NVVMReflectPass to the pipeline.
    machine->adjustPassManager(builder);
    // Add pass to initialize TTI for this specific target. Otherwise, TTI will
    // be initialized to NoTTIImpl by default.
    modulePM.add(
        createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));
    funcPM.add(
        createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));
  }

  builder.populateModulePassManager(modulePM);
  builder.populateFunctionPassManager(funcPM);
  funcPM.doInitialization();
  for (auto &func : *module) {
    funcPM.run(func);
  }
  funcPM.doFinalization();
  modulePM.run(*module);
}

void mlir::spn::CUDASerializeToCubinPass::linkWithLibdevice(
    llvm::Module *module) {
  // The kernel might use some optimized device functions from libdevice
  // (__nv_*, e.g. __nv_exp). libdevice is shipped as LLVM bitcode by Nvidia, so
  // we load the bitcode file and link it with the translated NVVM IR module.
  llvm::SMDiagnostic Err;
  auto libdevice =
      llvm::parseIRFile(SPNC_LIBDEVICE_FILE, Err, module->getContext());
  if (!libdevice) {
    SPNC_FATAL_ERROR("Failed to load libdevice: {}", Err.getMessage().str());
  }
  llvm::Linker::linkModules(*module, std::move(libdevice));

  // Internalize all functions except for the GPU kernel functions that were
  // present in the module before linking libdevice and conversion to NVVM.
  llvm::SmallSet<llvm::StringRef, 5> gpuKernels;
  gpuKernels.insert(kernelFuncs.begin(), kernelFuncs.end());
  llvm::internalizeModule(*module,
                          [&gpuKernels](const llvm::GlobalValue &V) -> bool {
                            if (gpuKernels.contains(V.getName())) {
                              return true;
                            }
                            return false;
                          });
}

std::unique_ptr<std::vector<char>>
mlir::spn::CUDASerializeToCubinPass::serializeISA(const std::string &isa) {
  // This code is mostly copy & pasta from SerializeToCubin.cpp

  // Text buffer to hold error messages if necessary.
  char jitErrorBuffer[4096] = {0};

  RETURN_ON_CUDA_ERROR(cuInit(0), "cuInit");

  // Linking requires a device context.
  CUdevice device = 0;
  RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context = nullptr;
  RETURN_ON_CUDA_ERROR(cuCtxCreate(&context, 0, device), "cuCtxCreate");
  CUlinkState linkState = nullptr;

  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void *jitOptionsVals[] = {jitErrorBuffer,
                            reinterpret_cast<void *>(sizeof(jitErrorBuffer))};

  RETURN_ON_CUDA_ERROR(cuLinkCreate(2,              /* number of jit options */
                                    jitOptions,     /* jit options */
                                    jitOptionsVals, /* jit option values */
                                    &linkState),
                       "cuLinkCreate");

  // Add the PTX assembly generated by LLVM's PTX backend to the link modules.
  auto kernelName = getOperation().getName().str();
  RETURN_ON_CUDA_ERROR(
      cuLinkAddData(linkState, CUjitInputType::CU_JIT_INPUT_PTX,
                    const_cast<void *>(static_cast<const void *>(isa.c_str())),
                    isa.length(), kernelName.c_str(), /* kernel name */
                    0,       /* number of jit options */
                    nullptr, /* jit options */
                    nullptr  /* jit option values */
                    ),
      "cuLinkAddData");

  void *cubinData = nullptr;
  size_t cubinSize = 0;
  RETURN_ON_CUDA_ERROR(cuLinkComplete(linkState, &cubinData, &cubinSize),
                       "cuLinkComplete");

  // Turn the generated Cubin into a binary blob that we can attach to the MLIR
  // host module.
  char *cubinAsChar = static_cast<char *>(cubinData);
  auto result =
      std::make_unique<std::vector<char>>(cubinAsChar, cubinAsChar + cubinSize);

  // This will also destroy the cubin data.
  RETURN_ON_CUDA_ERROR(cuLinkDestroy(linkState), "cuLinkDestroy");
  RETURN_ON_CUDA_ERROR(cuCtxDestroy(context), "cuCtxDestroy");

  return result;
}

mlir::spn::CUDASerializeToCubinPass::CUDASerializeToCubinPass(
    llvm::ArrayRef<llvm::StringRef> kernelFuctions, bool shouldPrintIR,
    unsigned _optLevel)
    : printIR{shouldPrintIR},
      kernelFuncs(kernelFuctions.begin(), kernelFuctions.end()),
      optLevel(_optLevel) {
  this->triple = "nvptx64-nvidia-cuda";
  this->chip = getGPUArchitecture();
  this->features = "+ptx60";
}

std::string mlir::spn::CUDASerializeToCubinPass::getGPUArchitecture() {
  // Text buffer to hold error messages if necessary.
  char jitErrorBuffer[4096] = {0};

  // Retrieve information about the compute capability of the GPU
  // hosted in this machine from the CUDA device driver API.
  // If multiple devices are present, arbitrarily choose the first
  // one to retrieve information from.

  RETURN_ON_CUDA_ERROR(cuInit(0), "cuInit");
  int numDevices = 0;
  RETURN_ON_CUDA_ERROR(cuDeviceGetCount(&numDevices), "cuDeviceGetCount");

  if (numDevices == 0) {
    SPDLOG_WARN("Found no CUDA devices, assuming architecture sm_35");
    return "sm_35";
  }
  SPDLOG_INFO("Found {} CUDA device(s)", numDevices);
  if (numDevices > 1) {
    SPDLOG_WARN("Found multiple CUDA devices, retrieving device information "
                "from first device");
  }
  CUdevice device = 0;
  RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context = nullptr;
  RETURN_ON_CUDA_ERROR(cuCtxCreate(&context, 0, device), "cuCtxCreate");
  char deviceName[1024] = {0};
  RETURN_ON_CUDA_ERROR(cuDeviceGetName(deviceName, sizeof(deviceName), device),
                       "cuDeviceGetName");
  SPDLOG_INFO("Querying GPU device {} for information", deviceName);
  int major = 0;
  RETURN_ON_CUDA_ERROR(
      cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                           device),
      "cuDeviceGetAttribute (Compute capability major)");
  int minor = 0;
  RETURN_ON_CUDA_ERROR(
      cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           device),
      "cuDeviceGetAttribute (Compute capability minor)");
  std::string arch = "sm_" + std::to_string(major) + std::to_string(minor);
  SPDLOG_INFO("GPU device {} supports compute capability {}", deviceName, arch);
  return arch;
}

std::unique_ptr<mlir::OperationPass<mlir::gpu::GPUModuleOp>>
mlir::spn::createSerializeToCubinPass(ArrayRef<llvm::StringRef> kernelFunctions,
                                      bool shouldPrintIR, unsigned optLevel) {
  return std::make_unique<CUDASerializeToCubinPass>(kernelFunctions,
                                                    shouldPrintIR, optLevel);
}