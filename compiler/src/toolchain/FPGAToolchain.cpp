#include "FPGAToolchain.h"

#include "pipeline/Pipeline.h"
#include "pipeline/BasicSteps.h"
#include "pipeline/steps/frontend/SPFlowToMLIRDeserializer.h"
#include "pipeline/steps/mlir/conversion/HiSPNtoLoSPNConversion.h"
#include "pipeline/steps/mlir/conversion/LoSPNtoCPUConversion.h"
#include "pipeline/steps/mlir/conversion/CPUtoLLVMConversion.h"
#include "pipeline/steps/mlir/conversion/MLIRtoLLVMIRConversion.h"
#include "pipeline/steps/mlir/transformation/LoSPNTransformations.h"
#include "pipeline/steps/codegen/EmitObjectCode.h"
#include "pipeline/steps/linker/ClangKernelLinking.h"
#include "TargetInformation.h"

using namespace spnc;
using namespace mlir;


namespace spnc {

std::unique_ptr<Pipeline<Kernel>> FPGAToolchain::setupPipeline(const std::string& inputFile,
                                                               std::unique_ptr<interface::Configuration> config)
{
  // Initialize the pipeline.
  std::unique_ptr<Pipeline<Kernel>> pipeline = std::make_unique<Pipeline<Kernel>>();

  // Initialize the MLIR context.
  auto ctx = std::make_unique<MLIRContext>();
  initializeMLIRContext(*ctx);

  // If IR should be dumped between steps/passes, we need to disable
  // multi-threading in MLIR
  if (spnc::option::dumpIR.get(*config)) {
    ctx->enableMultithreading(false);
  }
  auto diagHandler = setupDiagnosticHandler(ctx.get());
  // Attach MLIR context and diagnostics handler to pipeline context
  pipeline->getContext()->add(std::move(diagHandler));
  pipeline->getContext()->add(std::move(ctx));

  // Create an LLVM target machine and set the optimization level.
  int mcOptLevel = spnc::option::optLevel.get(*config);
  if (spnc::option::mcOptLevel.isPresent(*config) && spnc::option::mcOptLevel.get(*config) != mcOptLevel) {
    auto optionValue = spnc::option::mcOptLevel.get(*config);
    SPDLOG_INFO("Option mc-opt-level (value: {}) takes precedence over option opt-level (value: {})",
                optionValue, mcOptLevel);
    mcOptLevel = optionValue;
  }
  auto targetMachine = createTargetMachine(mcOptLevel);
  // Initialize kernel information.
  auto kernelInfo = std::make_unique<KernelInfo>();
  kernelInfo->target = KernelTarget::FPGA;
  // Attach the LLVM target machine and the kernel information to the pipeline context
  pipeline->getContext()->add(std::move(targetMachine));
  pipeline->getContext()->add(std::move(kernelInfo));

  // First step of the pipeline: Locate the input file.
  auto& locateInput = pipeline->emplaceStep < LocateFile < FileType::SPN_BINARY >> (inputFile);

  // Deserialize the SPFlow graph serialized via Cap'n Proto to MLIR.
  auto& deserialized = pipeline->emplaceStep<SPFlowToMLIRDeserializer>(locateInput);

  // Convert from HiSPN dialect to LoSPN.
  auto& hispn2lospn = pipeline->emplaceStep<HiSPNtoLoSPNConversion>(deserialized);

  // TODO: add lospn to fpga pass

  // TODO: This is not a tmp file! And arguments?
  auto& ipXactFile = pipeline->emplaceStep<CreateTmpFile<FileType::ZIP>>(true);
  //auto& emitIpXact = pipeline->emplaceStep<EmitObjectCode>(llvmConversion, ipXactFile);

  // Translate the generated LLVM IR module to object code and write it to an object file.
  auto& objectFile = pipeline->emplaceStep < CreateTmpFile < FileType::OBJECT >> (true);
  auto& emitObjectCode = pipeline->emplaceStep<EmitObjectCode>(llvmConversion, objectFile);

  // Link generated object file into shared object.
  auto& sharedObject = pipeline->emplaceStep < CreateTmpFile < FileType::SHARED_OBJECT >> (false);
  // Add additional libraries to the link command if necessary.
  llvm::SmallVector<std::string, 3> additionalLibs;

  // TODO: Instead of libraries we search for exteral verilog sources here.

  auto searchPaths = parseLibrarySearchPaths(spnc::option::searchPaths.get(*config));
  auto libraryInfo = std::make_unique<LibraryInfo>(additionalLibs, searchPaths);
  pipeline->getContext()->add(std::move(libraryInfo));
  // Link the kernel with the libraries to produce executable (shared object).
  (void) pipeline->emplaceStep<ClangKernelLinking>(emitObjectCode, sharedObject);
  // Add the CLI configuration to the pipeline context.
  pipeline->getContext()->add(std::move(config));

  return pipeline;
}

}