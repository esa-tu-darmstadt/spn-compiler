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
#include "pipeline/steps/hdl/CreateVivadoProject.h"
#include "pipeline/steps/hdl/EmbedController.hpp"
#include "pipeline/steps/hdl/EmbedAXIStream.hpp"
#include "pipeline/steps/hdl/WrapESI.hpp"
#include "pipeline/steps/hdl/ReturnKernel.h"
#include "pipeline/steps/hdl/WriteDebugInfo.hpp"
#include "pipeline/steps/hdl/CreateVerilogFiles.hpp"
#include "pipeline/steps/hdl/CreateAXIStreamMapper.hpp"
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
  // for debugging
  ctx->disableMultithreading();

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

  // First step of the pipeline: Locate the input file.
  auto& locateInput = pipeline->emplaceStep<LocateFile<FileType::SPN_BINARY>>(inputFile);

  // Deserialize the SPFlow graph serialized via Cap'n Proto to MLIR.
  auto& deserialized = pipeline->emplaceStep<SPFlowToMLIRDeserializer>(locateInput);

  // Convert from HiSPN dialect to LoSPN.
  auto& hispn2lospn = pipeline->emplaceStep<HiSPNtoLoSPNConversion>(deserialized);

  // does some preprocessing and fills KernelInfo with the correct information
  auto& lospnTransform = pipeline->emplaceStep<LoSPNTransformations>(hispn2lospn);

  // map the SPN operator to HW operators and perform scheduling
  std::string fpgaConfigJson = option::fpgaConfigJson.get(*config);
  auto& lospn2fpga = pipeline->emplaceStep<LoSPNtoFPGAConversion>(fpgaConfigJson, lospnTransform);

  // TODO: FIX THIS MESS!!!
  if (!option::justGetKernel.get(*config)) {
    //assert(option::fpgaWrapReadyValid.get(*config) || option::fpgaWrapAXIStream.get(*config) &&
    //  "Either option::fpgaWrapReadyValid or option::fpgaWrapAXIStream must be set to true!");

    //auto& embed2 = option::fpgaWrapReadyValid.get(*config) ?
    //  pipeline->emplaceStep<EmbedReadyValid>(lospn2fpga) :
    //  pipeline->emplaceStep<EmbedAXIStream>(lospn2fpga);

    if (option::fpgaWrapReadyValid.get(*config)) {
      auto& embed = pipeline->emplaceStep<EmbedReadyValid>(lospn2fpga);

      if (option::fpgaWrapESI.get(*config)) {
        assert(false && "not implemented");

        //auto& wrapESI = pipeline->emplaceStep<WrapESI>(
        //  embed, "ReadyValidWrapper",
        //  option::fpgaWrapESICosim.get(*config)
        //);
      }

      if (option::fpgaCreateVerilogFiles.get(*config)) {
        CreateVerilogFilesConfig cfg{
          .targetDir = option::outputPath.get(*config) + "/ipxact_core",
          .tmpdirName = "tmp",
          .topName = "ReadyValidWrapper"
        };

        auto& createVerilogFiles = pipeline->emplaceStep<CreateVerilogFiles>(embed, cfg);
        auto& writeDebugInfo = pipeline->emplaceStep<WriteDebugInfo>(
          spnc::option::outputPath.get(*config) + "/ipxact_core",
          fpgaConfigJson,
          createVerilogFiles
        );
      }
    } else if (option::fpgaWrapAXIStream.get(*config)) {
      auto& embed = pipeline->emplaceStep<EmbedAXIStream>(lospn2fpga);

      bool doPrepareForCocoTb = option::fpgaCocoTb.get(*config);
      auto& createAXIStreamMapper = pipeline->emplaceStep<CreateAXIStreamMapper>(embed, doPrepareForCocoTb);

      if (option::fpgaCreateVerilogFiles.get(*config)) {
        CreateVerilogFilesConfig cfg{
          .targetDir = option::outputPath.get(*config) + "/ipxact_core",
          .tmpdirName = "tmp",
          .topName = "AXIStreamWrapper"
        };

        VivadoProjectConfig vivadoConfig{
          .targetDir = cfg.targetDir
        };

        auto& createVerilogFiles = pipeline->emplaceStep<CreateVerilogFiles>(createAXIStreamMapper, cfg);
        auto& createVivadoProject = pipeline->emplaceStep<CreateVivadoProject>(createVerilogFiles, vivadoConfig);
        auto& writeDebugInfo = pipeline->emplaceStep<WriteDebugInfo>(
          spnc::option::outputPath.get(*config) + "/ipxact_core",
          fpgaConfigJson,
          createVivadoProject
        );
      }
    } else {
      assert(false && "not implemented");
      //auto& embedController = pipeline->emplaceStep<EmbedAXIStream>(lospn2fpga);
      //auto& wrapESI = pipeline->emplaceStep<WrapESI>(embedController, "AXIStreamWrapper", false);
    }
  } else {
    auto& returnKernel = pipeline->emplaceStep<ReturnKernel>();
    auto& writeDebugInfo = pipeline->emplaceStep<WriteDebugInfo>(
      spnc::option::outputPath.get(*config) + "/ipxact_core",
      fpgaConfigJson,
      returnKernel
    );
  }

  // Attach the LLVM target machine and the kernel information to the pipeline context
  pipeline->getContext()->add(std::move(targetMachine));
  pipeline->getContext()->add(std::move(kernelInfo));
  pipeline->getContext()->add(std::move(config));

  return pipeline;
}

}