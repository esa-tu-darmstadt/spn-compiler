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
#include "pipeline/steps/hdl/EmitVerilogCode.h"
#include "pipeline/steps/hdl/CreateIPXACT.h"
#include "pipeline/steps/mlir/conversion/LoSPNtoFPGAConversion.h"
#include "TargetInformation.h"

using namespace spnc;
using namespace mlir;


namespace spnc {

std::unique_ptr<Pipeline<Kernel>> FPGAToolchain::setupPipeline(const std::string& inputFile,
                                                               std::unique_ptr<interface::Configuration> config)
{
  llvm::outs() << "FPGAToolchain::setupPipeline() called\n";

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

  auto targetMachine = createTargetMachine(0);
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

  auto& lospn2fpga = pipeline->emplaceStep<LoSPNtoFPGAConversion>(hispn2lospn);

  auto& emitVerilogCode = pipeline->emplaceStep<EmitVerilogCode>(lospn2fpga);

  IPXACTConfig ipConfig{
    .sourceFilePaths = {
      "resources/ufloat/FPOps_build_add/FPAdd.v",
      "resources/ufloat/FPOps_build_mult/FPMult.v",
      "resources/ufloat/FPLog.v"
    },
    .targetDir = "./ipxact_core",
    .topModuleFileName = "spn_body.v"
  };
  auto& createIPXACT = pipeline->emplaceStep<CreateIPXACT>(ipConfig, emitVerilogCode);

  pipeline->getContext()->add(std::move(config));

  return pipeline;
}

}