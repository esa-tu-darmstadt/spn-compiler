#include "LoSPNtoFPGAConversion.h"

#include "LoSPN/Analysis/SPNBitWidth.h"
#include "LoSPNtoFPGA/LoSPNtoFPGAPass.h"
#include "LoSPNtoFPGA2/LoSPNtoFPGAPass2.hpp"
#include "toolchain/MLIRToolchain.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/HW/HWOps.h"

#include <filesystem>
#include <sstream>
#include <util/json.hpp>
#include <Kernel.h>


namespace spnc {

void LoSPNtoFPGAConversion::preProcess(mlir::ModuleOp *inputModule) {
  namespace fs = std::filesystem;
  using json = nlohmann::json;

  if (fpgaConfigJson.empty()) {
    spdlog::info("No fpga-config-json file specified!");
    return;
  }

  std::string jsonText;

  if (fs::is_regular_file(fpgaConfigJson)) {
    spdlog::info("Attempting to parse fpga-config-json from file {}", fpgaConfigJson);
    std::ifstream jsonFile(fpgaConfigJson);
    assert(jsonFile.is_open() && "Could not open fpga-config-json file");
    std::stringstream buffer;
    buffer << jsonFile.rdbuf();
    jsonText = buffer.str();
  } else {
    spdlog::info("Attempting to parse fpga-config-json from command line argument");
    jsonText = fpgaConfigJson;
  }

  getContext()->add<Kernel>(std::move(Kernel{FPGAKernel{}}));
  FPGAKernel& kernel = getContext()->get<Kernel>()->getFPGAKernel();

  // Don't ask me what happens when an invalid json is passed. Exceptions are disabled Q_Q
  json j = json::parse(jsonText);

  kernel.memDataWidth = j.at("axi4").at("dataWidth");
  kernel.memAddrWidth = j.at("axi4").at("addrWidth");

  kernel.liteDataWidth = j.at("axi4Lite").at("dataWidth");
  kernel.liteAddrWidth = j.at("axi4Lite").at("addrWidth");

  kernel.projectName = j.at("projectName");
  kernel.kernelId = j.at("kernelId");
  kernel.deviceName = j.at("device").at("name");
  kernel.deviceSpeed = j.at("device").at("mhz");

  std::string floatType = j.at("floatType");

  if (floatType == "float32")
    kernel.spnResultWidth = 32;
  else if (floatType == "float64")
    kernel.spnResultWidth = 64;
  else
    assert(false && "Invalid floatType in fpga-config-json");

  ::mlir::spn::SPNBitWidth bitWidth(inputModule->getOperation());

  auto kernelInfo = getContext()->get<KernelInfo>();
  kernel.spnVarCount = kernelInfo->numFeatures;
  kernel.spnBitsPerVar =
    bitWidth.getBitsPerVar() == 2 ? 2 : roundN<uint32_t>(bitWidth.getBitsPerVar(), 8);

  kernel.mAxisControllerWidth = roundN(kernel.spnResultWidth, 8);
  kernel.sAxisControllerWidth = roundN(kernel.spnBitsPerVar * kernel.spnVarCount, 8);
}

void LoSPNtoFPGAConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  mlir::spn::fpga::ConversionOptions conversionOptions{
    .ufloatConfig = {
      .exponentWidth = floatExponentWidth,
      .mantissaWidth = floatMantissaWidth
    },
    .use32Bit = use32Bit
  };

  pm->addPass(
    mlir::spn::fpga::createLoSPNtoFPGAPass2(conversionOptions)
  );

  //struct LowerSeqFIRRTLToSVOptions {
  //  bool disableRegRandomization = false;
  //  bool addVivadoRAMAddressConflictSynthesisBugWorkaround = false;
  //};
  circt::seq::LowerSeqFIRRTLToSVOptions options;

  pm->nest<circt::hw::HWModuleOp>().addPass(
    circt::seq::createSeqFIRRTLLowerToSVPass(options)
  );

}

void LoSPNtoFPGAConversion::postProcess(mlir::ModuleOp *resultModule) {
  FPGAKernel& kernel = getContext()->get<Kernel>()->getFPGAKernel();

  resultModule->walk([&](mlir::Operation *op){
    if (!op->hasAttr("fpga.body_delay"))
      return;

    kernel.bodyDelay = llvm::dyn_cast<mlir::IntegerAttr>(
      op->getAttr("fpga.body_delay")
    ).getInt();
  });

  kernel.fifoDepth = kernel.bodyDelay * 2;

  spdlog::info(kernel.to_string());
}

}